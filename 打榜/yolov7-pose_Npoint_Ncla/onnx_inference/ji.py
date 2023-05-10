import json
import logging as log
import numpy as np
import cv2
import onnxruntime
import torch
from torchvision import transforms


model_class_names = ["stand", "sit", "crouch", "prostrate_sleep", "sit_sleep", "lie_sleep"]  ###模型榜，需要检测的类别名称
alert_class_names = ["prostrate_sleep", "sit_sleep", "lie_sleep"]  ###算法榜，需要检测的类别名称
#device = torch.device("cuda:0")


def init():
    """Initialize model

    Returns: model

    """
    model_path = "/project/train/models/best.onnx"

    return model_path


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
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
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def read_img(img):
    image = img
    image = letterbox(image, 320, stride=32, auto=False)[0]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    # if torch.cuda.is_available():
    #     image = image.to(device)

    return image


def model_inference(model_path=None, input=None):
    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name
    output = session.run([], {input_name: input.numpy()})
    return output


def process_image(model_path, input_image, args=None):
    input = read_img(input_image)
    output = model_inference(model_path, input)
    result_json = post_process(output[0], input_image)

    return result_json


def post_process(output, img, score_threshold=0.3):
    """
    Draw bounding boxes on the input image. Dump boxes in a txt file.
    """
    nc = 6
    target_info, objects = [], []
    det_bboxes, det_scores, det_labels, kpts = output[:, 0:4], output[:, 4], output[:, 5], output[:, 5+nc:]
    for idx in range(len(det_bboxes)):
        det_bbox = det_bboxes[idx]
        kpt = kpts[idx]
        ######################################
        kpt_score = np.sum(kpt[2::3]) / 17
        kpt[2::3] = np.round(kpt[2::3])
        kpt[0::3] *= img.shape[1] / 320
        kpt[1::3] *= img.shape[0] / 320
        det_bbox[0::2] *= img.shape[1] / 320
        det_bbox[1::2] *= img.shape[0] / 320
        name = model_class_names[int(det_labels[idx])]
        conf = det_scores[idx]
        ######################################
        if det_scores[idx] > score_threshold:
            x = int(det_bbox[0])  # xmin
            y = int(det_bbox[1])  # ymin
            width = int(det_bbox[2] - det_bbox[0])  # width
            height = int(det_bbox[3] - det_bbox[1])  # height
        #######################################
            if name in model_class_names:
                obj = {'x': x, 'y': y, 'width': width, 'height': height, 'confidence': float(conf), 'name': name,
                       ###检测框用x,y,width,height表示, 也可以
                       'keypoints': {'keypoints': kpt.tolist(), 'score': kpt_score}}
                objects.append(obj)
            if name in alert_class_names:
                alert_obj = {'x': x, 'y': y, 'width': width, 'height': height, 'confidence': float(conf), 'name': name,
                             ###检测框用x,y,width,height表示, 也可以
                             'keypoints': {'keypoints': kpt.tolist(), 'score': kpt_score}}
                target_info.append(alert_obj)

    target_count = len(target_info)
    is_alert = True if target_count > 0 else False
    return json.dumps(
        {'algorithm_data': {'is_alert': is_alert, 'target_count': target_count, 'target_info': target_info},
         'model_data': {"objects": objects}})


if __name__ == '__main__':
    """Test python api
    """
    img = cv2.imread("/home/data/*/ZDSsleeping20230329_V2_sample_factory_in_100_11512.jpg")
    model_path = init()
    result = process_image(model_path, img)
    log.info(result)
