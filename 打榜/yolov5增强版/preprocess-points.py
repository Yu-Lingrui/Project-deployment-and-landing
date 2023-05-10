import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import cv2
import logging
from glob import glob
import time
import random

'''
YOLO v5 
xml -> txt
'''


classes = ['banner']
class2id = {name:i for i, name in enumerate(classes)}

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])

    x1 = (box[0] + box[2])/2.0 - 1
    y1 = (box[1] + box[7])/2.0 - 1
    w1 = abs(box[2] - box[0])
    h1 = abs(box[7] - box[1])
    x2 = (box[4] + box[6])/2.0 - 1
    y2 = (box[3] + box[5])/2.0 - 1
    w2 = abs(box[4] - box[6])
    h2 = abs(box[5] - box[3])
    x = (x1+x2)/2.0
    w = (w1+w2)/2.0
    y = (y1+y2)/2.0
    h = (h1+h2)/2.0
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    x = min(x, 1.0)
    y = min(y, 1.0)
    w = min(w, 1.0)
    h = min(h, 1.0)
    
    return (x,y,w,h)
 
def convert_annotation(image_path):
    in_file = open(image_path.replace('.xml', '.xml'),encoding="utf-8")
    out_file = open(image_path.replace('.xml', '.txt'), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    if w == 0 or h == 0: 
        print(1)
        return
    for obj in root.iter('ploygon'):
        name = obj.find('class').text
        cls_id = class2id[name]
        xmlbox = obj.find('points')
        b = xmlbox.text.split(';')
        bb = (float(b[0].split(',')[0]),float(b[0].split(',')[1]),float(b[1].split(',')[0]),float(b[1].split(',')[1]),
             float(b[2].split(',')[0]),float(b[2].split(',')[1]),float(b[3].split(',')[0]),float(b[3].split(',')[0]))
        bbb = convert((w,h), b,)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bbb]) + '\n')

# 1. 转换数据label
files = glob('/home/data/*/*.xml')
for file in files:
    convert_annotation(file)

# 2. 划分train与valid ~ K折
K = 5
files = glob('/home/data/*/*.txt')
random.shuffle(files)
ind = len(files) // 5
# train = [x.replace('.txt', '.jpg')+'\n' for x in files[ind:]]
# valid = [x.replace('.txt', '.jpg')+'\n' for x in files[:ind]]
train = [x.replace('.txt', '.jpg') for x in files[ind:]]
valid = [x.replace('.txt', '.jpg') for x in files[:ind]]
print(f"train {len(train)}, valid {len(valid)}")

# 3. 写入文件
with open('train.txt', 'w') as f:
    f.write('\n'.join(train))
with open('valid.txt', 'w') as f:
    f.write('\n'.join(valid))
