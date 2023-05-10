#!/bin/bash

project_root_dir=/project/train/src_repo
log_file=/project/train/log/log.txt

pip install ultralytics \
&& echo "Preparing..." \
&& echo "Converting dataset..." \
&& python3 -u ${project_root_dir}/convert_dataset.py | tee -a ${log_file} \
&& echo "Start training..." \
&& cd ${project_root_dir} && yolo pose train data=mydata.yaml model=yolov8s-pose.pt pretrained=True device=0 project=/project/train/models epochs=200 imgsz=640 cache=True optimizer=SGD cache batch=16 iou=0.5 | tee -a ${log_file} \
&& echo "Export onnx..." \
&& yolo export model=/project/train/models/train/weights/best.pt data=mydata.yaml format=onnx simplify=True opset=13 \
&& echo "Copy result-graphs" \
&& cp /project/train/models/train/results.csv /project/train/tensorboard/ \
&& cp /project/train/models/train/*.png /project/train/tensorboard/ \
&& cp /project/train/models/train/results.png /project/train/result-graphs/ \
&& echo "Done"

