from ultralytics import YOLO
import cv2

model = YOLO("yolov8s-pose.pt")

# from ndarray
im2 = cv2.imread("1.jpg")
im2 = cv2.resize(im2, (640,640))
results = model.predict(source=im2)  # save predictions as labels

res_plotted = results[0].plot()
cv2.imshow("result", res_plotted)
cv2.waitKey(-1)