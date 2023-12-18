from ultralytics import YOLO

# Load a model
# model = YOLO('mydata.yaml')  # build a new model from YAML
# model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8s.yaml').load('./yolov8s.pt')  # build from YAML and transfer weights
model = YOLO('/project/train/models/exp/weights/last.pt')
# Train the model
# model.train(data='mydata.yaml', epochs=100, imgsz=480, cache=True,batch=40, device=0, workers=4, name="exp",optimizer="AdamW",lr0=1E-3,project="/project/train/models",cos_lr=True,resume=True) # 480
model.train(data='mydata.yaml', epochs=100, imgsz=640, cache=True,batch=40, device=0, workers=4, name="exp",optimizer="AdamW",lr0=1E-3,project="/project/train/models",cos_lr=True,resume=True) # 640