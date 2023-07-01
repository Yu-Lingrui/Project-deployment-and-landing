#!/bin/bash

project_root_dir=/project/train/src_repo
log_file=/project/train/log/log.txt

# pip install ultralytics \
echo "Preparing..." \
&& echo "Converting dataset..." \
&& python3 -u ${project_root_dir}/convert_dataset.py | tee -a ${log_file} \
&& echo "Start training..." \
&& cd ${project_root_dir} && yolo pose train data=mydata.yaml model=yolov8l-pose.pt pretrained=True device=0 project=/project/train/models epochs=300 imgsz=640 cache=True optimizer=SGD batch=16 iou=0.5 | tee -a ${log_file} \
&& echo "Export onnx..." \
# && yolo export model=/project/train/models/train/weights/best.pt data=mydata.yaml format=onnx simplify=True opset=13 \
&& yolo export model=/project/train/models/train/weights/best.pt data=mydata.yaml format=onnx simplify=True opset=13 imgsz=(192, 320) \
&& echo "Transform onnx..." \
&& python /project/train/src_repo/v8_transform.py /project/train/models/train/weights/best.onnx
# && echo "Copy result-graphs" \
# && cp /project/train/models/train/results.csv /project/train/tensorboard/ \
# && cp /project/train/models/train/*.png /project/train/tensorboard/ \
# && cp /project/train/models/train/results.png /project/train/result-graphs/ \
# && echo "Done"

