yolov5：单输入单输出，输入1*3*size*size, 输出 1*n*(5+class),输出包含坐标还原不包含nms
yolov7：单输入单输出，输入1*3*size*size, 输出 1*n*(5+class),输出包含坐标还原不包含nms
yolox：官方源代码


 atc  --model /usr/local/ev_model_ability_supply/model/exported_plate_with_type_0222.onnx --framework 5  --output exported_plate_with_type_0222 --soc_version Ascend310
 target exported_plate_with_type_0222.om


 atc  --model /usr/local/ev_model_ability_supply/model/yolov5s.onnx --framework 5  --output yolov5s --soc_version Ascend310
 target yolov5s.om


 atc  --model /usr/local/ev_model_ability_supply/model/yolov7.onnx --framework 5  --output yolov7 --soc_version Ascend310
 target yolov7.om


 atc  --model /usr/local/ev_model_ability_supply/model/yolox.onnx --framework 5  --output yolox --soc_version Ascend310
 target yolox.om