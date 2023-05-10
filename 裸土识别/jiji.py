weight_pth = '/project/train/models/model_last.pth'
n_cat = 4
device = 'cuda:0'
imgsize = [480, 480]

import sys
sys.path.insert(0, '/project/train/src_repo/BiSeNet/')
import json
import torch
import cv2
import numpy as np
import os
from lib.models import model_factory
import torch.nn.functional as F

def init():
    """Initialize model
    Returns: model
    """
    global weight_path
    net = model_factory['bisenetv2'](n_cat)
    net.load_state_dict(torch.load(weight_pth, map_location='cpu'))
    net.cuda()
    net.eval()
    return net

def process_image(handle=None,input_image=None,args=None, **kwargs):

    """Do inference to analysis input_image and get output
    Attributes:
        handle: algorithm handle returned by init()
        input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
        args: string in JSON format, format: {
            "mask_output_path": "/path/to/output/mask.png"
        }
    Returns: process result
    """
    global device, imgsize
    
    args =json.loads(args)
    mask_output_path =args['mask_output_path']
    h, w, _ = input_image.shape
    
    # Convert
    img = img0.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float().to('cuda')
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    # Normalization
    mean = (0.46962251, 0.4464104,  0.40718787)
    std = (0.27469736, 0.27012361, 0.28515933)
    img_mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
    img_std = np.array(std, dtype=np.float32).reshape(3, 1, 1)
    img_mean = torch.tensor(img_mean).float().cuda()
    img_std = torch.tensor(img_std).float().cuda()
    img = (img - img_mean) / img_std
    
    if len(img.shape) == 3:
        img = img[None] 
    pred = handle(img)[0]
    pred = F.interpolate(pred, size=(h, w),
        mode='bilinear', align_corners=True)
    pred = pred.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype('uint8')
    cv2.imwrite(mask_output_path, pred)
    return json.dumps({'mask': mask_output_path}, indent=4)

if __name__ == '__main__':
    # Test API
    from glob import glob
    img = cv2.imread(glob('/home/data/*/*.jpg')[0])
    predictor = init()
    res = process_image(predictor, img, "{\"mask_output_path\":\"./out.jpg\"}")
    print(res)
    