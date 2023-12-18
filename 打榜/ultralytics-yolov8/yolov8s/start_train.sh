#!/bin/bash

project_root_dir=/project/train/src_repo
log_file=/project/train/log/log.txt

# pip install ultralytics \
echo "Preparing..." \
&& echo "Converting dataset..." \
&& python3 -u ${project_root_dir}/convert_dataset.py | tee -a ${log_file} \
&& echo "Start training..." \
&& cd ${project_root_dir} && python train.py | tee -a ${log_file} 
# && cp /project/train/models/exp/*.png /project/train/result-graphs/ \
# && rm /project/train/models/exp/*.png \
# && cp /project/train/models/exp/*.jpg /project/train/result-graphs/ \
# && rm /project/train/models/exp/*.jpg \
# && cp /project/train/models/exp/*.csv /project/train/result-graphs/ \
# && rm /project/train/models/exp/*.csv 
#&& echo "Export onnx..." \
# && yolo export model=/project/train/models/train/weights/best.pt data=mydata.yaml format=onnx simplify=True opset=13 \
#&& yolo export model=/project/train/models/train/weights/best.pt data=mydata.yaml format=onnx simplify=True opset=13 imgsz=(192, 320) \
#&& echo "Transform onnx..." \
#&& python /project/train/src_repo/v8_transform.py /project/train/models/train/weights/best.onnx
# && echo "Copy result-graphs" \
# && cp /project/train/models/train/results.csv /project/train/tensorboard/ \
# && cp /project/train/models/train/*.png /project/train/tensorboard/ \
# && cp /project/train/models/train/results.png /project/train/result-graphs/ \
# && echo "Done"

