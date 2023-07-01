import os
import shutil
import time
from pathlib import Path

import cv2
import torch
from numpy import random
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import time_sync, select_device
from utils.datasets import LoadStreams, LoadImages
from utils.plots import Annotator, colors
from backend.flask_id2name import id2name

def predict(opt, model, img):
    out, source, view_img, save_img, save_txt, imgsz = \
        opt['output'], opt['source'], opt['view_img'], opt['save_img'], opt['save_txt'], opt['imgsz']

    #webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    device = select_device(opt['device']) # 选择设备
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    im0_shape = img.shape # 记下原始图片的尺寸
    #print('im0_shape = %s \n' % str(im0_shape))

    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages(opt['source'], img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0) # batch 1

        # Inference
        t1 = time_sync()
        
        # 前向推理
        pred = model(img, augment=opt['augment'])[0] 
        # Apply NMS（非极大抑制）
        pred = non_max_suppression(pred, opt['conf_thres'], opt['iou_thres'], classes=opt['classes'], agnostic=opt['agnostic_nms'])
       
        t2 = time_sync()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s
            save_path = str(Path(out) / Path(p).name) # 保存路径
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                boxes_detected = [] #检测结果
                for *xyxy, conf, cls in reversed(det):
                    xyxy_list = (torch.tensor(xyxy).view(1, 4)).view(-1).tolist()  
                    boxes_detected.append({"name": id2name[int(cls.item())],
                                    "conf": str(conf.item()),
                                    "bbox": [int(xyxy_list[0]), int(xyxy_list[1]), int(xyxy_list[2]), int(xyxy_list[3])]
                                    })
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        line_thickness = 3
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)

    results = {"results": boxes_detected}
    print(results)
    return results