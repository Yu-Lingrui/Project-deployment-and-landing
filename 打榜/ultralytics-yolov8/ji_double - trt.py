import os
import sys
import cv2
import glob
import json
import time
from tqdm import tqdm
import numpy as np

sys.path.append("/project/train/src_repo/")
import torch
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from models.experimental import attempt_load

########################################trt##################################
import logging
import os
import onnxruntime
import torchvision
import tensorrt as trt
from collections import OrderedDict, namedtuple
from pathlib import Path


logger = trt.Logger(trt.Logger.INFO)
LOGGER = logging.getLogger("yolov5")
##################################################################################

CONF_THRESH = 0.5 # 0.3
HEAD_CONF_THRESH = 0.5 # 0.4
IOU_THRESHOLD = 0.5
WIDTH_THRESHOLD = 0.0
HEIGHT_THRESHOLD = 0.2
device = 'cuda:0'
# model_mt_path = '/usr/local/ev_sdk/model/exp/weights/best.pt'
model_mt_path = 'best.onnx'
# model_mt_path = '/project/train/models/exp/weights/yolov5s.pt'
# model_head_path = '/usr/local/ev_sdk/model/head.pt'
model_head_path = 'last.onnx'
# model_head_path = '/project/train/models/exp/weights/yolov5s.pt'
imgsz_mt = 480 # 960
imgsz_head = 480
stride_mt = None
stride_head = None
half = True

# categories = ['head1',"hat","helmet","electric_scooter_person","motorbike_person"]
categories = ['head']
# target_categories = ['head1',"helmet","hat","electric_scooter_person","motorbike_person"] 
target_categories = ['head']

#############################################################################################################################
def export_trt(model_path):
    f = Path(model_path).with_suffix('.engine')  # TensorRT engine file
    onnx = f.with_suffix(".onnx")
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    workspace = 4
    config.max_workspace_size = workspace * 1 << 30
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f'failed to load ONNX file: {onnx}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f' input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f' output "{out.name}" with shape{out.shape} {out.dtype}')

    # if builder.platform_has_fast_fp16 and half:
    #     config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())
    return str(f)
                     
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
 
class yolo():
    def __init__(self, model_path, device = torch.device('cpu'), confThreshold=0.2, iouThreshold=0.5, objThreshold=0.5):
        import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
        if device.type == 'cpu':
            device = torch.device('cuda:0')
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        if model_path.endswith(".onnx"):
            model_path = export_trt(model_path)
        logger = trt.Logger(trt.Logger.INFO)
        with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        self.context = model.create_execution_context()
        self.bindings = OrderedDict()
        fp16 = False  # default updated below
        dynamic = False
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            if model.binding_is_input(index):
                if -1 in tuple(model.get_binding_shape(index)):  # dynamic
                    dynamic = True
                    self.context.set_binding_shape(index, tuple(model.get_profile_shape(0, index)[2]))
                if dtype == np.float16:
                    fp16 = True
            shape = tuple(self.context.get_binding_shape(index))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.img_size = [480,480]
        self.stride = 32
        self.confThresh = confThreshold
        self.iouThresh = iouThreshold

    def from_numpy(self, x):
        return torch.from_numpy(x).to("cpu") if isinstance(x, np.ndarray) else x

    def detect(self, img):
        im = letterbox(img, self.img_size, stride=self.stride, auto=False)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to("cuda")
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]

        # im = im.cpu().numpy()  # torch to numpy
        s = self.bindings['images'].shape
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = self.bindings['output'].data
        if isinstance(y, (list, tuple)):
            prediction = self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            prediction = self.from_numpy(y)
        bs = prediction.shape[0]
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > self.confThresh  # candidates
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence
            x[:, 5:] *= x[:, 4:5]
            box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > self.confThresh]
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                return [], im
            x = x[x[:, 4].argsort(descending=True)]
            boxes, scores = x[:, :4] , x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, self.iouThresh)

            return x[i], im
####################################################################################################### 
                     
def convert_results(result_boxes, result_scores, result_classid):
    detect_objs = []
    global hat_count
    for j in range(len(result_boxes)):
        box = result_boxes[j]
        x0, y0, x1, y1 = box
        conf = result_scores[j]
        detect_objs.append({
            'x': int(x0),
            'y': int(y0),
            'width': int(x1-x0),
            'height': int(y1-y0),
            'confidence': float(conf),
            'name': categories[int(result_classid[j])]
        })
    
    target_objs = []
    for obj in detect_objs:
        if obj["name"] in target_categories:
            target_objs.append(obj)

    res = {
        "algorithm_data": {
            "is_alert": len(target_objs) > 0,
            "target_count": len(target_objs),
            "target_info": target_objs
        },
        "model_data": {
            "objects": detect_objs
        }
    }

    return json.dumps(res, indent=4)


@torch.no_grad()
def init():
    
    model_mt_path = "best.onnx"
    model_head_path = "last.onnx"     
    session_mt = yolo(model_mt_path)
    session_head = yolo(model_head_path)
#     global imgsz_mt, imgsz_head, stride_mt,stride_head, device, half

#     # Load model
#     device = select_device(device)
#     # model = DetectMultiBackend(model_path, device=device, dnn=False)
#     model_mt = attempt_load(model_mt_path, map_location=device)
    
    
#     stride_mt = int(model_mt.stride.max())
#     imgsz_mt = check_img_size(imgsz_mt, s=stride_mt)  # check image size
#     model_head = attempt_load(model_head_path, map_location=device)
#     stride_head = int(model_head.stride.max())
#     imgsz_head = check_img_size(imgsz_head, s=stride_head)  # check image size
#     # Half
#     model_mt.half() if half else model_mt.float()
#     model_head.half() if half else model_head.float()
    return [session_mt,session_head]

@torch.no_grad()
def process_image(model, input_image, args=None, **kwargs):
    global result_boxes
    global result_scores
    global result_classid
    # img = letterbox(input_image, imgsz_mt, stride=stride_mt, auto=True)[0]  # BGR
    # # tc('letterbox')
    # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB 
    # img = np.expand_dims(img, axis=0)
    # img = np.ascontiguousarray(img)
    # img = torch.from_numpy(img).to(device)
    # img = img.half() if half else img.float()  # uint8 to fp16/32
    # img /= 255  # 0 - 255 to 0.0 - 1.0
    # # tc('255')
    # pred = model[0](img, augment=False, visualize=False)[0]
    # # tc('infer')
    # conf_thres = CONF_THRESH  # confidence threshold
    # iou_thres = IOU_THRESHOLD  # NMS IOU threshold
    # classes = None  # filter by class: --class 0, or --class 0 2 3
    # agnostic_nms = False  # class-agnostic NMS
    # max_det = 1000  # maximum detections per image
    # pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    # # tc('nms')
    # det = pred[0]
    # output_image = input_image
    net = model[0]                 
    det, resizeIm = net.detect(input_image)
    if len(det):
        det[:, :4] = scale_coords(resizeIm.shape[2:], det[:, :4], input_image.shape).round()
        result_boxes, result_scores, result_classid = predict_head(input_image, det[:, :4], model[1])
        # 合并预测结果
        # result_boxes_mt =  det[:, :4]
        # result_scores_mt = det[:, 4]
        # result_classid_mt = det[:, 5]+3
        # result_boxes = torch.cat((result_boxes,result_boxes_mt),dim=0)
        # result_scores = torch.cat((result_scores, result_scores_mt ),dim=0)
        # result_classid = torch.cat((result_classid,  result_classid_mt), dim=0)
        # result_boxes, result_scores, result_classid = det[:, :4], det[:, 4], det[:, 5]
    else:
        result_boxes, result_scores, result_classid = [], [], []
    # tc('post')
    res = convert_results(result_boxes, result_scores, result_classid)
    # tc('convert')
    return res
def predict_head(input_image, boxes, model):
    global result_boxes
    global result_scores 
    global result_classid
    img_batch = []
    img_shape = []
    # 裁剪打包batch
    # print(f"batch size：{boxes.shape[0]}")
    # print(boxes[:,2]-boxes[:,0])
    # imgsz = (imgsz_head, torch.max(boxes[:,2]-boxes[:,0]).cpu().item())
    # print(f"imgsz_head: {imgsz}")
    
    for i in range(boxes.shape[0]):
        x0, y0, x1, y1 = boxes[i]
        #  处理box 
        w = WIDTH_THRESHOLD*(x1-x0)
        h = HEIGHT_THRESHOLD*(y1-y0)
        x0 = max(0, x0-w)
        y0 = max(0, y0-h)
        x1 = min(input_image.shape[1], x1+w)
        y1 = min(input_image.shape[0], y1)
        boxes[i,0],boxes[i,1],boxes[i,2],boxes[i,3] = x0, y0, x1, y1
        ## 处理box结束
        img = input_image[int(y0):int(y1),int(x0):int(x1),:]
        # cv2.rectangle(img, (int(w+3/8*width), int(h)),(int(w+5/8*width), int(h+1/5*height)), (0, 255, 0))
        img_shape.append(img.shape)
        
        # img = letterbox(img, imgsz_head, stride=stride_head, auto=False)[0]  # BGR
        # print(f"img shape: {img.shape}")
        img_batch.append(img)
    # img_batch = np.array(img_batch)
    # # print(f"img_batch shape: {img_batch.shape}")
    # img_batch = img_batch.transpose((0, 3, 1, 2))[:,::-1,:,:]  # HWC to CHW, BGR to RGB
    # img_batch = np.ascontiguousarray(img_batch)
    # img_batch = torch.from_numpy(img_batch).to(device)
    # img_batch = img_batch.half() if half else img_batch.float()  # uint8 to fp16/32
    # img_batch /= 255  # 0 - 255 to 0.0 - 1.0
        # img = np.expand_dims(img, axis=0)   
    # 预测
    # pred = model(img_batch, augment=False, visualize=False)[0]
    # conf_thres = HEAD_CONF_THRESH  # confidence threshold
    # iou_thres = IOU_THRESHOLD  # NMS IOU threshold
    # classes = None  # filter by class: --class 0, or --class 0 2 3
    # agnostic_nms = False  # class-agnostic NMS
    # max_det = 1000  # maximum detections per image
    # pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    # result_boxes = torch.tensor([]).to(device)
    # result_scores = torch.tensor([]).to(device)
    # result_classid = torch.tensor([]).to(device)
    # 解析
    # print(f"检测结果数量：{len(pred)}") 
    pred = []
    global resizeIm
    for img in img_batch:            
        pre, resizeIm = model.detect(img)
        pred.append(pre)
    for i in range(len(pred)):
        det = pred[i]
        
        x0, y0, _, _ = boxes[i]
        if len(det):
            det[:, :4] = scale_coords(resizeIm.shape[2:], det[:, :4], img_shape[i]).round()
            det[:, 0] += x0
            det[:, 1] += y0
            det[:, 2] += x0
            det[:, 3] += y0
            result_boxes = det[:, :4]
            # result_boxes = torch.cat((result_boxes,det[:, :4]))
            # print(result_boxes.shape[0])
            result_scores = det[:, 4]
            # result_scores = torch.cat((result_scores,det[:, 4]))

            # print(result_scores.shape[0])
            result_classid = result_classid,det[:, 5]
            # result_classid = torch.cat((result_classid,det[:, 5]))

            # print(result_classid.shape[0])
            # print(f"boxes nums： {len(result_boxes)}")
#     画图
#     output_image = input_image
#     for box,conf,cls_id in zip(result_boxes,result_scores,result_classid):
#         x_min, y_min, x_max, y_max = box
#         cls_conf = f"conf: {conf} class：{categories[cls_id]}"
#         cv2.rectangle(output_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)  # 画矩形框
#         cv2.putText(output_image, str(cls_conf), (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)  # 添加类别文本
        
    return result_boxes, result_scores, result_classid

if __name__ == '__main__':
    # model_path = '/project/train/models/exp/weights/best.pt'
    # imgsz = 640
    # Test API

    # img_paths = glob.glob("/home/data/*/*.jpg")
    img_paths = glob.glob("1.jpg")
    model = init()

    total_time = 0
      
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        # start = time.time()
        # tc.init()  
        result = process_image(model, img)
        # save_path = img_path.replace("inputs", "outputs")
        # cv2.imwrite(save_path, output_image)
        print(result)
        # end = time.time()
        # total_time += end - start
    # tc.summury()

    # print(">> time >> " + str(total_time))
    # print(">> fps >> " + str(len(img_paths) / total_time))
