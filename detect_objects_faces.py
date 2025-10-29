import cv2
import numpy as np
import face_recognition
import os

video_path = "video/sample.mp4"
cap = cv2.VideoCapture(video_path)

# Load YOLO
net = cv2.dnn.readNet("models/yolo/yolov3.weights", "models/yolo/yolov3.cfg")
with open("models/yolo/coco.names") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load known faces
known_encodings, known_names = [], []
for filename in os.listdir("models/faces/"):
    if filename.endswith(".jpg"):
        image = face_recognition.load_image_file(f"models/faces/{filename}")
        enc = face_recognition.face_encodings(image)[0]
        known_encodings.append(enc)
        known_names.append(os.path.splitext(filename)[0])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(detection[0] * width - w / 2)
                y = int(detection[1] * height - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    # Face detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locs = face_recognition.face_locations(rgb_frame)
    face_encs = face_recognition.face_encodings(rgb_frame, face_locs)

    for (top, right, bottom, left), face_encoding in zip(face_locs, face_encs):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            name = known_names[matches.index(True)]
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Objects & Faces Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
