import os
import cv2
import numpy as np
from gtts import gTTS
from matplotlib.pyplot import close
from playsound import playsound
#from torch import classes


net=cv2.dnn.readNet("yolov3.weights","darknet-master\darknet-master\cfg\yolov3.cfg")
classes=[]
with open("darknet-master\darknet-master\data\coco.names","r") as f:
    classes=[line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))
# Loading image
vedio_capture=cv2.VideoCapture(0)
o=0
while True:
    #img = cv2.imread("Screenshot_2022-02-13-23-20-57-99_1c337646f29875672b5a61192b9010f9.jpg")
    _,img=vedio_capture.read()

    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
        # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
        # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
                
            if confidence > 0.5:
                    # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                    # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                    

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    l=[]
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
            l.append(label)
            
    txt=""
    for i in l:
        txt=txt+i+" "


    tts=gTTS(text=txt,lang='en',slow=False)
    ti="ab"+str(o)+".mp3"
    tts.save(ti)
    playsound(ti)
    o+=1
        

        #os.system("start hello.mp3")


    cv2.imshow("Image", img)
    key=cv2.waitKey(2)
    if key & 0xFF==ord('q'):
        break

cv2.destroyAllWindows()
