from glob import glob
import random
import json

'''
YOLO v7 pose
xml -> txt
'''

classes = ["stand", "sit", "crouch", "prostrate_sleep", "sit_sleep", "lie_sleep"]
class2id = {name: i for i, name in enumerate(classes)}


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])

    xmin = box[0]
    ymin = box[1]
    xmax = box[0] + box[2]
    ymax = box[1] + box[3]
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2
    w = box[2]
    h = box[3]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    x = min(x, 1.0)
    y = min(y, 1.0)
    w = min(w, 1.0)
    h = min(h, 1.0)

    return x, y, w, h


def convert_annotation(jsonpath):
    f = open(jsonpath, 'rb')
    out_file = open(jsonpath.replace('.json', '.txt'), 'w')
    infos = json.load(f)
    for info in infos['annotations']:
        xmin, ymin, width, height = info['bbox']  ###检测框的左上角坐标和高宽
        box_name = info['category_name']  ###检测框的名称
        cls_id = class2id[box_name]
        w = int(info['width'])
        h = int(info['height'])
        b = (float(xmin), float(ymin), float(width), float(height))
        bb = convert((w, h), b, )
        points_nor_list = []
        for point in info['keypoints']:
            points_nor_list.append(float(point))
        del points_nor_list[2::3]  # 删除从第三个元素开始的每第三个元素
        points_nor_list = [x / h if i % 2 else x / w for i, x in enumerate(points_nor_list)]
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + " " +
                       " ".join([str(c) for c in points_nor_list]) + '\n')


# 1. 转换数据label
files = glob('/home/data/*/*.json')
for file in files:
    convert_annotation(file)

# 2. 划分train与valid ~ K折
K = 9
files = glob('/home/data/*/*.txt')
random.shuffle(files)
ind = len(files) // K
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
