from ultralytics import YOLO

model = YOLO('best.pt')
model.export(format='engine', device=0, simplify=True, half=True)