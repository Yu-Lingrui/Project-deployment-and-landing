echo "Export onnx..." \
&& yolo export model=/project/train/models/train/weights/best.pt data=mydata.yaml format=onnx simplify=True opset=13 \
&& echo "Transform onnx..." \
&& python /project/train/src_repo/v8_transform.py /project/train/models/train/weights/best.onnx