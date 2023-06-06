from ultralytics import YOLO
model=YOLO("yolov8n.pt")
model.predict(source="img6.jpg",show=True)