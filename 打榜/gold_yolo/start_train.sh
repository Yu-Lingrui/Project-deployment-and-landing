#!/bin/bash

project_root_dir=/project/train/src_repo/Efficient-Computing/Detection/Gold-YOLO
log_file=/project/train/log/log.txt
tb_path=/project/train/tensorboard
plt_path=/project/train/models/exp
echo "Preparing..." \
&& echo "Converting dataset..." \
&& python3 -u /project/train/src_repo/convert_dataset.py | tee -a ${log_file} \
&& echo "Start training..." \
&& cd ${project_root_dir} && python3 -u tools/train.py --batch 32 --conf configs/gold_yolo-s.py --data ${project_root_dir}/data/dataset.yaml --epoch 200 --fuse_ab --device 0 | tee -a ${log_file} \
&& echo "Export onnx..." \
&& cp /project/train/models/exp/weights/best_ckpt.pt /project/train/models/head.pt \
&& python deploy/ONNX/export_onnx.py --weights /project/train/models/exp/weights/best_ckpt.pt --device 0 --simplify \
&& cp /project/train/models/exp/weights/best_ckpt.onnx /project/train/models/head.onnx \
&& echo "Done" 
# && echo "Remove result-graphs file" \
# && rm  ${plt_path}/* \
# && echo "Copy font..." \
# && cp ${project_root_dir}/Arial.ttf /project/.config/Ultralytics/Arial.ttf | tee -a ${log_file} \
