import os
import cv2
import face_recognition
import csv
import numpy as np
from gtts import gTTS
from matplotlib.pyplot import close
from playsound import playsound
#from torch import classes
img=face_recognition.load_image_file("photos\elon.jpg")
imgenc=face_recognition.face_encodings(img)[0]

img_1=face_recognition.load_image_file("photos\jefff.jpg")
imgenc_1=face_recognition.face_encodings(img_1)[0]

img_2=face_recognition.load_image_file("photos\kkkkk.jpg")
imgenc_2=face_recognition.face_encodings(img_1)[0]

img_3=face_recognition.load_image_file("photos\kris.jpg")
imgenc_3=face_recognition.face_encodings(img_3)[0]

img_4=face_recognition.load_image_file("photos\kash.jpeg")
imgenc_4=face_recognition.face_encodings(img_4)[0]

img_5=face_recognition.load_image_file("photos\kash1.jpeg")
imgenc_5=face_recognition.face_encodings(img_5)[0]

img_6=face_recognition.load_image_file("photos\kash2.jpeg")
imgenc_6=face_recognition.face_encodings(img_6)[0]




know_face_encode=[imgenc,imgenc_1,imgenc_2,imgenc_3,imgenc_4,imgenc_5,imgenc_6]
know_face_name=["elon","jeff","kris","kris","akash","akash","akash"]

studen=know_face_name.copy()

face_locations=[]
face_encodings=[]
face_names=[]
s=True


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
    try:
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
                if label=="person":
                    continue
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                l.append(label)
        if s:
            frame=img
            #smalframe=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
            smalframe=img
            rgb_smalfram=smalframe[:,:,::-1]
            face_locations=face_recognition.face_locations(rgb_smalfram)
            face_encodings=face_recognition.face_encodings(rgb_smalfram,face_locations)
            face_names=[]
            for i in face_encodings:
                match=face_recognition.compare_faces(know_face_encode,i)
                name="Unknown"
                face_dist=face_recognition.face_distance(know_face_encode,i)
                bestmatch=np.argmin(face_dist)
                if match[bestmatch]:
                    name=know_face_name[bestmatch]
                face_names.append(name)
        s=not s
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            l.append(name)
                
        txt=""
        for i in l:
            txt=txt+i+" "


        tts=gTTS(text=txt,lang='en',slow=False)
        ti="ab"+str(o)+".mp3"
        tts.save(ti)
        
        
            

            #os.system("start hello.mp3")


        cv2.imshow("Image", img)
        playsound(ti)
        key=cv2.waitKey(0)
        if key & 0xFF==ord('q'):
            break
    except:
         tts=gTTS(text="picture is not clear due to dim light",lang='en',slow=False)
         ti="ab"+str(o)+".mp3"
         tts.save(ti)
         playsound(ti)
    o+=1

cv2.destroyAllWindows()
