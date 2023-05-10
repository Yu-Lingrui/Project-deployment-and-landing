import json
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from linkNet import *
def to_image(img, min_side):
    size = img.shape
    h, w = size[0], size[1]
    #长边缩放为min_side
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w/scale), int(h/scale)
    resize_img = cv2.resize(img, (new_w, new_h))
    # 填充至min_side * min_side
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2, (min_side-new_h)/2, (min_side-new_w)/2 + 1, (min_side-new_w)/2
    elif new_h % 2 != 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2 + 1, (min_side-new_h)/2, (min_side-new_w)/2, (min_side-new_w)/2
    elif new_h % 2 == 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2, (min_side-new_h)/2, (min_side-new_w)/2, (min_side-new_w)/2
    else:
        top, bottom, left, right = (min_side-new_h)/2 + 1, (min_side-new_h)/2, (min_side-new_w)/2 + 1, (min_side-new_w)/2
    pad_img = cv2.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=[0,0,0]) #从图像边界向上,下,左,右扩的像素数目
    return pad_img,int(top), int(bottom), int(left), int(right)
def rev_to_image(img,h,w,top,bottom,left,right):
    resize_img = cv2.resize(img[top:512-bottom,left:512-right], (h, w))
    return resize_img
def init():
    model = LinkNet(2)
    model.load_state_dict(torch.load("/project/train/models/model.pth"))
    model = model.cuda()
    model.eval()
    input = torch.randn(1, 3, 512, 512).cuda()
    with torch.no_grad():
        for _ in range(100):
            model(input)
    return model

def process_image(handle=None, input_image=None, args=None, **kwargs):
    args =json.loads(args)
    mask_output_path =args['mask_output_path']
    model=handle
    h, w, _ = input_image.shape
    img=input_image
    img,top,bottom,left,right=to_image(img,512)
    img = np.transpose(img, (2, 0, 1))
    img = img / 255.
    img = torch.from_numpy(img).reshape(1, 3, 512, 512)
    img = img.type(torch.FloatTensor).cuda()
    prd = model(img).cpu().detach().numpy() 
    prd = np.asarray(np.argmax(prd, axis=1), dtype=np.uint8)
    prd=np.reshape(prd,(1,512,512))  
    prd = np.transpose(prd, (1,2,0))
    prd=np.reshape(prd,(512,512)).astype('float32')  
    
    prd = cv2.resize(prd[top:512-bottom,left:512-right], (w, h))     
    cv2.imwrite(mask_output_path, prd)
    return json.dumps({
        "mask": mask_output_path,
        "algorithm_data": {},

        "model_data": {

            "mask": mask_output_path

        }

    }, indent=4)

if __name__ == '__main__':
    """Test python api
    """
    img = cv2.imread('/home/data/1430/square_20220815_v1_p_train_street_1_225.jpg')
    # img = np.random.randn(1024,512,3)
    model = init()
    process_image(model,img,json.dumps({"mask_output_path": "1.png"}, indent=4))