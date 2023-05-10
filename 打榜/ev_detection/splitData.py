import glob
import os.path
import random


train_split_rate = 0.8 ##  !!! 注意可修改此处
###  0---------------------------------以下不用修改-----------------------------------------
dataPath = "/home/data/*/*.jpg"
imgList = glob.glob(dataPath)
random.shuffle(imgList)
trainList = imgList[:int(train_split_rate*len(imgList))]
testList = imgList[int(train_split_rate*len(imgList)):]

if os.path.exists("train.txt"):
    os.remove("train.txt")
with open("train.txt", "w") as f:
    for i in trainList:
        f.write(i)
        f.write("\n")


if os.path.exists("test.txt"):
    os.remove("test.txt")
with open("test.txt", "w") as f:
    for i in testList:
        f.write(i)
        f.write("\n")
