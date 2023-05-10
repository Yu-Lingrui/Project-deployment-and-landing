#!/bin/bash
cd /project/train/src_repo/yolov7-pose_Npoint_Ncla
#rm -rf /project/train/models/*
#pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

python preprocess.py

python train_Ncla_nPoint.py --batch-size 16  --epochs 100 --project ../../tensorboard --img-size 480

python models/export.py --weights /project/train/tensorboard/*/weights/best.pt --simplify --device 0 --img-size 480
mv /project/train/tensorboard/*/weights/* /project/train/models/
rm -rf /project/train/tensorboard/*/weights
#python -m onnxsim demobilized_weights/best.onnx demobilized_weights/best_sim.onnx
