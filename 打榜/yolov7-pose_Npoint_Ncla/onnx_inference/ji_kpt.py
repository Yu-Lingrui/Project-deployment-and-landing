import logging as log
import read
import os
# os.system('pip install -i https://mirrors.aliyun.com/pypi/simple onnxruntime')
# os.system('pip install -i https://mirrors.aliyun.com/pypi/simple onnxruntime-gpu==1.7.0')
import onnxruntime as ort
import cv2
import numpy as np
from keypoint_postprocess import HRNetPostProcess
os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
log.basicConfig(level=log.DEBUG)

class yolov5():
    def __init__(self, model_path, name_path, confThreshold=0.5, nmsThreshold=0.5, objThreshold=0.5):
        self.classes = list(map(lambda x: x.strip(), open(name_path, 'r').readlines()))
        self.colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(self.classes))]
        num_classes = len(self.classes)
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.no = num_classes + 5
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        self.inpWidth = 640
        self.inpHeight = 640
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession(model_path, so)
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold
        self.keep_ratio = True
    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)
    def resize_image(self, srcimg):
        top, left, newh, neww = 0, 0, self.inpHeight, self.inpWidth
        if self.keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.inpWidth - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.inpWidth - neww - left, cv2.BORDER_CONSTANT,
                                         value=0)  # add border
            else:
                newh, neww = int(self.inpHeight * hw_scale), self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.inpHeight - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.inpHeight - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left
    def detect(self, srcimg):
        img, newh, neww, top, left = self.resize_image(srcimg)
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32)
        blob /= 255.0
        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0].squeeze(axis=0)

        # inference output
        row_ind = 0
        for i in range(self.nl):
            h, w = int(self.inpHeight/self.stride[i]), int(self.inpWidth/self.stride[i])
            length = int(self.na * h * w)
            if self.grid[i].shape[2:4] != (h,w):
                self.grid[i] = self._make_grid(w, h)

            outs[row_ind:row_ind+length, 0:2] = (outs[row_ind:row_ind+length, 0:2] * 2. - 0.5 + np.tile(self.grid[i],(self.na, 1))) * int(self.stride[i])
            outs[row_ind:row_ind+length, 2:4] = (outs[row_ind:row_ind+length, 2:4] * 2) ** 2 * np.repeat(self.anchor_grid[i],h*w, axis=0)
            row_ind += length

        srcimgHeight = srcimg.shape[0]
        srcimgWidth = srcimg.shape[1]
        ratioh, ratiow = srcimgHeight / newh, srcimgWidth / neww
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for detection in outs:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if detection[4] > self.objThreshold:
                center_x = (detection[0] - left) * ratiow
                center_y = (detection[1] - top) * ratioh
                width = detection[2] * ratiow
                height = detection[3] * ratioh
                xmin = int(center_x - width * 0.5)
                ymin = int(center_y - height * 0.5)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([xmin, ymin, int(width), int(height)])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        if not isinstance(indices, (tuple, list)):
            indices = indices.flatten()
        results = []
        for i in indices:
            box = boxes[i]
            results.append((max(box[0], 0), max(box[1], 0), min(box[0] + box[2], srcimgWidth) , min(box[1] + box[3], srcimgHeight), self.classes[classIds[i]], confidences[i]))
        return results

class KeyPointDetector():
    def __init__(self, modelpath):
        self.class_names = ["keypoint"]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(modelpath, so)
        self.inpWidth = self.session.get_inputs()[0].shape[3]
        self.inpHeight = self.session.get_inputs()[0].shape[2]
        self.imshape = np.array([[self.inpWidth, self.inpHeight]], dtype=np.float32)
        self.visual_thresh = 0.5
        self.use_dark = False

    def resize_image(self, srcimg, keep_ratio=True):
        padh, padw, newh, neww = 0, 0, self.inpWidth, self.inpHeight
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_LINEAR)
                padw = int((self.inpWidth - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, padw, self.inpWidth - neww - padw, cv2.BORDER_CONSTANT, value=0)  # add border
            else:
                newh, neww = int(self.inpHeight * hw_scale), self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_LINEAR)
                padh = int((self.inpHeight - newh) * 0.5)
                img = cv2.copyMakeBorder(img, padh, self.inpHeight - newh - padh, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_LINEAR)
        return img, newh, neww, padh, padw

    def predict(self, img, box_info):
        img, newh, neww, padh, padw = self.resize_image(img)
        # try:
        #     img, newh, neww, padh, padw = self.resize_image(img)
        # except:
        #     return [{'keypoints':np.zeros(17*3).tolist(), 'score':0.5}]
        
        img = np.transpose(img.astype(np.float32), [2, 0, 1])
        result = self.session.run(None, {'image': img[None, :, :, :]})


        center = np.round(self.imshape / 2.)
        scale = self.imshape / 200.
        keypoint_postprocess = HRNetPostProcess(use_dark=self.use_dark)
        kpts, scores = keypoint_postprocess(result[0], center, scale)
        for i in range(kpts.shape[1]):
            kpts[0, i, 0] = box_info['xmin'] + (kpts[0, i, 0] - padw) * (
                    box_info['xmax'] - box_info['xmin']) / neww
            kpts[0, i, 1] = box_info['ymin'] + (kpts[0, i, 1] - padh) * (
                    box_info['ymax'] - box_info['ymin']) / newh

        results = []
        # if kpts.shape[0]>17:
        #     kpts = kpts[:17, :, :]
        for i in range(kpts.shape[0]):
            kpts[i, :, 2][kpts[i, :, 2] < self.visual_thresh] = 0
            kpts[i, :, 2][kpts[i, :, 2] >= self.visual_thresh] = 1
            results.append({'keypoints':kpts[i,:,:].flatten().tolist(), 'score':float(scores[i,0])})
            # results.append({'keypoints': kpts[i, :, :].flatten().tolist()})
        return results

kpt_predictor = KeyPointDetector("/usr/local/ev_sdk/model/dark_hrnet_w32_256x192.onnx")

def init():
    """Initialize model

    Returns: model

    """
    model_path = "/project/train/models/best.onnx"
    name_path = '/project/train/models/label.txt'
    if not os.path.isfile(model_path):
        log.error(f'{model_path} does not exist')
        return None
    log.info('Loading model...')
    model = yolov5(model_path, name_path)
    return model

def process_image(net, input_image, args=None):
    results = net.detect(input_image)
    detect_objs = []
    for k, det in enumerate(results):
        xmin, ymin, xmax, ymax, name, score = det
        img = input_image[ymin:ymax, xmin:xmax]
        
        kpts = kpt_predictor.predict(img, {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
        obj = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'confidence':float(score), 'name':name, 'keypoints':kpts[0]}
        detect_objs.append(obj)
        
    return read.dumps({'model_data': {"objects": detect_objs}})


if __name__ == '__main__':
    """Test python api
    """
    img = cv2.imread('/usr/local/ev_sdk/data/persons.jpg')
    predictor = init()
    result = process_image(predictor, img)
    log.info(result)

