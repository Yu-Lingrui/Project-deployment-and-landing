import os
import shutil
from os import listdir, getcwd
from os.path import join
import random
import argparse
import xml.etree.ElementTree as ET
classes= ['person','head','door_open','door_half_open','door_close','daizi','bottle','bag','box','plastic_basket','suitcase','mobile_phone','umbrella','folder','bicycle','electric_scooter']   #自己训练的类别
datasets_path = '/home/data/'  # 新划分数据集地址
data_root_path = '/home/data/*/'  # 原始数据集地址


# 将jpg和xml分离
def jpg_xml():
    if not os.path.exists(datasets_path + 'Annotations/'):
        os.makedirs(datasets_path + 'Annotations/')
    if not os.path.exists(datasets_path + 'images/'):
        os.makedirs(datasets_path + 'images/')
    filelist = os.listdir(data_root_path)
    for files in filelist:
        filename1 = os.path.splitext(files)[1]  # 读取文件后缀名
        if filename1 == '.jpg':
            full_path = os.path.join(data_root_path, files)
            shutil.move(full_path, datasets_path+'images')
        elif filename1 == '.xml':
            full_path = os.path.join(data_root_path, files)
            shutil.move(full_path, datasets_path+'Annotations')
        else :
            continue


# 划分数据集
def train_val_split(): #运行此函数
    trainval_percent = 0.2
    train_percent = 0.8

    images_filepath = datasets_path + 'images/'
    txtsavepath = datasets_path
    total_imgfiles = os.listdir(images_filepath)

    num = len(total_imgfiles)
    lists = range(num)

    tr = int(num * train_percent)
    train = random.sample(lists, tr)

    ftrain = open(txtsavepath + 'train.txt', 'w+')
    ftest = open(txtsavepath +  'test.txt', 'w+')
    fval = open(txtsavepath + 'val.txt', 'w+')

    for i in lists:
        name = images_filepath + total_imgfiles[i] + '\n'
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
            ftest.write(name)

    ftrain.close()
    fval.close()
    ftest.close()


# 转换box
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


# 转换标签
def convert_annotation(image_id):
    in_file = open(datasets_path + 'Annotations/%s.xml' % (image_id),encoding='utf-8')
    out_file = open(datasets_path + 'labels/%s.txt' % (image_id), 'w',encoding='utf-8')

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = 0
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b=(float(xmlbox.find('xmin').text),float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
        float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


# 生成新的标签
def generate_labels():  # 运行此函数
    if not os.path.exists(datasets_path + 'labels/'):
        os.makedirs(datasets_path + 'labels/')
    sets = ['train', 'val']
    for image_set in sets:
        image_ids = open(datasets_path + '%s.txt' % (image_set)).read().strip().split('\n')
        for image_id in image_ids:
            convert_annotation(image_id.split('/')[-1][:-4])


if __name__ == "__main__":

    jpg_xml()
    train_val_split()
    generate_labels()