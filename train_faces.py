import face_recognition
import cv2
import numpy as np
import csv
from gtts import gTTS

from playsound import playsound
import os

vedio_capture=cv2.VideoCapture(0)
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

while True:
    ret,frame=vedio_capture.read()
    
    if s:
        smalframe=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
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
        tts=gTTS(text=name,lang='en',slow=False)
        tts.save('hello.mp3')
        playsound('hello.mp3')
    cv2.imshow("vedio",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vedio_capture.release()
cv2.destroyAllWindows()
