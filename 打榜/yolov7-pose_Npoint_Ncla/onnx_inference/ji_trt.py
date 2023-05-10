import json
import logging
import os
import onnxruntime
import cv2
import numpy as np
import torch
import torchvision
import tensorrt as trt
from collections import OrderedDict, namedtuple
from pathlib import Path
import time


logger = trt.Logger(trt.Logger.INFO)
LOGGER = logging.getLogger("yolov5")

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

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


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


class yolo():
    def __init__(self, model_path, name_path, device = torch.device('cpu'), confThreshold=0.5, iouThreshold=0.5, objThreshold=0.5):
        self.classes = list(map(lambda x: x.strip(), open(name_path, 'r').readlines()))
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
        self.img_size = [640,640]
        self.stride = 32
        self.confThresh = confThreshold
        self.iouThresh = iouThreshold

    def getClassName(self):
        return self.classes

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


def init():
    """Initialize model

    Returns: model

    """
    # model_path = "/usr/local/ev_sdk/model/best.onnx"
    name_path = '/usr/local/ev_sdk/src/labels.txt'
    model_path = "/project/train/models/exp/weights/best.onnx"
    session = yolo(model_path, name_path)
    return session


def process_image(net, input_image, args=None):
    det, resizeIm = net.detect(input_image)
    className = net.getClassName()
    detect_objs = []
    if len(det):
        det[:, :4] = scale_coords(resizeIm.shape[2:], det[:, :4], input_image.shape).round()
        for *xyxy, conf, cls in reversed(det):
            detect_objs.append({
                "name": className[int(cls)],
                "xmin": int(xyxy[0]),
                "ymin": int(xyxy[1]),
                "xmax": int(xyxy[2]),
                "ymax": int(xyxy[3]),
                'confidence': float(conf)
            })

    #         cv2.rectangle(input_image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), thickness=1)
    #         cv2.putText(input_image, str(className[int(cls)]) + ': ' + str(round(float(conf), 3)), (int(xyxy[0]), int(xyxy[1]) - 10),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)

    # cv2.imwrite('result.jpg', input_image)
    return json.dumps({'model_data':{"objects": detect_objs}})


if __name__ == '__main__':
    """Test python api
    """
    img = cv2.imread(
       "/home/data/10078/SXphotovoltaic20220704_V1_train_factory_out_1_001027.jpg")
    predictor = init()
    result = process_image(predictor, img)
    log.info(result)
