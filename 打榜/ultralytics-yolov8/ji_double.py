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

CONF_THRESH = 0.5 # 0.3
HEAD_CONF_THRESH = 0.5 # 0.4
IOU_THRESHOLD = 0.5
WIDTH_THRESHOLD = 0.0
HEIGHT_THRESHOLD = 0.2
device = 'cuda:0'
model_mt_path = '/usr/local/ev_sdk/model/exp/weights/best.pt'
# model_mt_path = '/project/train/models/exp/weights/yolov5s.pt'
model_head_path = '/usr/local/ev_sdk/model/head.pt'
# model_head_path = '/project/train/models/exp/weights/yolov5s.pt'
imgsz_mt = 640 # 960
imgsz_head = 480
stride_mt = None
stride_head = None
half = True

# categories = ['head',"hat","helmet","electric_scooter_person","motorbike_person"]
categories = ['head1',"hat","helmet"]
# target_categories = ['head',"helmet","hat","electric_scooter_person","motorbike_person"] 
target_categories = ['head1',"hat"] 


def convert_results(result_boxes, result_scores, result_classid):
    detect_objs = []
    global hat_count
    for j in range(len(result_boxes)):
        box = result_boxes[j]
        x0, y0, x1, y1 = box
        conf = result_scores[j]
        # detect_objs.append({
        #     'xmin': int(x0),
        #     'ymin': int(y0),
        #     'xmax': int(x1),
        #     'ymax': int(y1),
        #     'confidence': float(conf),
        #     'name': categories[int(result_classid[j])]
        # })
        detect_objs.append({
            'x': int(x0),
            'y': int(y0),
            'width': int(x1-x0),
            'height': int(y1-y0),
            'confidence': float(conf),
            'name': categories[int(result_classid[j])]
        })
        # if int(result_classid[j])==3:
        #     detect_objs.append({
        #     'x': int(x0),
        #     'y': int(y0),
        #     'width': int(x1-x0),
        #     'height': int(y1-y0),
        #     'confidence': float(conf),
        #     'name': categories[int(result_classid[j])+1]
        #     })

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

# if __name__ == "__main__":
#     class time_count:
#         def __init__(self):
#             self.count = 0
#             self.count1_call_times = 0
#             self.names = []

#         def init(self):
#             self.count += 1
#             self.count_call_times = 0

#             if self.count == 1:
#                 pass
#             elif self.count == 2:
#                 self.list = [[] for _ in range(self.count1_call_times)]
#             elif self.count > 2:
#                 pass
#             else:
#                 raise Exception("time_count")
#             self.start = time.time()

#         def __call__(self, name):
#             self.end = time.time()
#             if self.count == 1:
#                 self.count1_call_times += 1
#                 self.names.append(name)
#             elif self.count > 1:
#                 self.list[self.count_call_times].append(self.end - self.start)
#                 self.count_call_times += 1
#             else:
#                 raise Exception("time_count")
#             self.start = time.time()

#         def summury(self):
#             print("==" * 20)
#             spend_time_mean_overall = 0
#             for idx in range(len(self.names)):
#                 name = self.names[idx]
#                 spend_time_list = self.list[idx]
#                 spend_time_mean = np.mean(spend_time_list) * 1000
#                 spend_time_std = np.std(spend_time_list) * 1000
#                 if spend_time_mean > 0.01:
#                     print("{: <80}{: <20}{: <20}".format(name, round(spend_time_mean, 2), round(spend_time_std, 10)))
#                 spend_time_mean_overall += spend_time_mean
#             print("{: <80}{: <20}".format("overall_time", round(spend_time_mean_overall, 1)))
#             print("{: <80}{: <20}".format("overall_fps", round(1000 / spend_time_mean_overall, 1)))
#             print("==" * 20)
# else:
#     class time_count:
#         def __init__(self):
#             pass

#         def init(self):
#             pass

#         def __call__(self, name):
#             pass

#         def summury(self):
#             pass
# global tc
# tc = time_count()

@torch.no_grad()
def init():

    global imgsz_mt, imgsz_head, stride_mt,stride_head, device, half

    # Load model
    device = select_device(device)
    # model = DetectMultiBackend(model_path, device=device, dnn=False)
    model_mt = attempt_load(model_mt_path, map_location=device)
    
    
    stride_mt = int(model_mt.stride.max())
    imgsz_mt = check_img_size(imgsz_mt, s=stride_mt)  # check image size
    model_head = attempt_load(model_head_path, map_location=device)
    stride_head = int(model_head.stride.max())
    imgsz_head = check_img_size(imgsz_head, s=stride_head)  # check image size
    # Half
    model_mt.half() if half else model_mt.float()
    model_head.half() if half else model_head.float()
    return [model_mt,model_head]

@torch.no_grad()
def process_image(model, input_image, args=None, **kwargs):
    img = letterbox(input_image, imgsz_mt, stride=stride_mt, auto=True)[0]  # BGR
    #############################################################################
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR to RGB
    #############################################################################
    # tc('letterbox')
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB x 翻转
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    # tc('255')
    pred = model[0](img, augment=False, visualize=False)[0]
    # tc('infer')
    conf_thres = CONF_THRESH  # confidence threshold
    iou_thres = IOU_THRESHOLD  # NMS IOU threshold
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS
    max_det = 1000  # maximum detections per image
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    # tc('nms')
    det = pred[0]
    # output_image = input_image
    if len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], input_image.shape).round()
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
        cv2.rectangle(img, (int(w+3/8*width), int(h)),(int(w+5/8*width), int(h+1/5*height)), (0, 255, 0))
        img_shape.append(img.shape)
        
        img = letterbox(img, imgsz_head, stride=stride_head, auto=False)[0]  # BGR
        # print(f"img shape: {img.shape}")
        img_batch.append(img)
    img_batch = np.array(img_batch)
    # print(f"img_batch shape: {img_batch.shape}")
    img_batch = img_batch.transpose((0, 3, 1, 2))[:,::-1,:,:]  # HWC to CHW, BGR to RGB
    img_batch = np.ascontiguousarray(img_batch)
    img_batch = torch.from_numpy(img_batch).to(device)
    img_batch = img_batch.half() if half else img_batch.float()  # uint8 to fp16/32
    img_batch /= 255  # 0 - 255 to 0.0 - 1.0
        # img = np.expand_dims(img, axis=0)   
    # 预测
    pred = model(img_batch, augment=False, visualize=False)[0]
    conf_thres = HEAD_CONF_THRESH  # confidence threshold
    iou_thres = IOU_THRESHOLD  # NMS IOU threshold
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS
    max_det = 1000  # maximum detections per image
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    result_boxes = torch.tensor([]).to(device)
    result_scores = torch.tensor([]).to(device)
    result_classid = torch.tensor([]).to(device)
    # 解析
    # print(f"检测结果数量：{len(pred)}")
    for i in range(len(pred)):
        det = pred[i]
        
        x0, y0, _, _ = boxes[i]
        if len(det):
            det[:, :4] = scale_coords(img_batch.shape[2:], det[:, :4], img_shape[i]).round()
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
    img_paths = glob.glob("/project/inputs/*.jpg")
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
