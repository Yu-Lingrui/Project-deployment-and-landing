import os
from glob import glob
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn  
from linkNet import LinkNet
def process_image(img, min_side):
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
class DatasetFolder(Dataset):
    def __init__(self):
        self.train_list = []
        files = glob('/project/train/src_repo/trainval/*.jpg')
        for line in files:
            self.train_list.append(line)
    def __getitem__(self, index):
        item=self.train_list[index]
        bg_item = item[:-4] + '.png'
        img = cv2.imread(item)
        label = cv2.imread(bg_item,cv2.IMREAD_GRAYSCALE)
        img,top,bottom,left,right=process_image(img,512)
        label,top,bottom,left,right=process_image(label,512)
        img=np.transpose(img,(2, 0, 1))
        img = img / 255.
        img = torch.from_numpy(img)
        label =torch.from_numpy(label)
        return img, label
    def __len__(self):
        return len(self.train_list)
path="/project/train/models/model.pth"
net=LinkNet(2)
# net.load_state_dict(torch.load(path, map_location='cpu'))
net.cuda()
criterion=nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
epochs= 400
train_dataset = DatasetFolder()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
losses=[]
print("training...")
max_score=0.
for epoch in range(0, epochs):
    net.train()
    for i, (input, lable) in enumerate(train_loader):
        input = input.type(torch.FloatTensor).cuda()
        lable = lable.type(torch.LongTensor).cuda()
        output = net(input)
        # print(np.shape(output))
        loss = criterion(output, lable)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.6f}\t mean_loss {mean_loss:.6f}'.format(
                epoch, i, len(train_loader), loss=loss.item()*1e5,mean_loss=np.mean(losses[-100:])*1e5)) 
            torch.save(net.state_dict(), "/project/train/models/model.pth")
            print("model save!!!!")