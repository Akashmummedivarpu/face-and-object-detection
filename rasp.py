import os
import cv2
import face_recognition
import csv
import numpy as np
from gtts import gTTS
import torch
from matplotlib.pyplot import close
from playsound import playsound
import pytesseract
from googletrans import Translator
from PIL import Image

pytesseract.pytesseract.tesseract_cmd=r'C:/Program Files/Tesseract-OCR/tesseract.exe'

img_0=face_recognition.load_image_file("photos\khil.jpg")
imgenc_0=face_recognition.face_encodings(img_0)[0]

#img_1=face_recognition.load_image_file("photos\khil1.jpg")
#imgenc_1=face_recognition.face_encodings(img_1)[0]

img_2=face_recognition.load_image_file("photos\pavan2.jpg")
imgenc_2=face_recognition.face_encodings(img_2)[0]

img_3=face_recognition.load_image_file("photos\pavan1.jpg")
imgenc_3=face_recognition.face_encodings(img_3)[0]

img_4=face_recognition.load_image_file("photos\kash.jpeg")
imgenc_4=face_recognition.face_encodings(img_4)[0]

img_5=face_recognition.load_image_file("photos\kash1.jpeg")
imgenc_5=face_recognition.face_encodings(img_5)[0]

img_6=face_recognition.load_image_file("photos\kash2.jpeg")
imgenc_6=face_recognition.face_encodings(img_6)[0]

know_face_encode=[imgenc_0,imgenc_2,imgenc_3,imgenc_4,imgenc_5,imgenc_6]
know_face_name=["akhil","pavan","pavan","akash","akash","akash"]

studen=know_face_name.copy()

face_locations=[]
face_encodings=[]
face_names=[]
s=True
classes=[]

img=cv2.imread("photos\pavan1.jpg")
model=torch.hub.load("WongKinYiu/yolov7","yolov7")

tts=gTTS(text="to convert given image and extract text press tts otherwise press any key to decetect object",lang="en",slow=False)
tts.save('hel.mp3')
print("tts")
playsound('hel.mp3')
selection=input()
d=dict()
tts=gTTS(text="please select your language telugu press 1 for hindi press 2 for english press 3 for tamil press 4 for malyalam press 5",lang='en',slow=False)
tts.save('hello.mp3')
print("lang")
playsound('hello.mp3')
a=int(input())

d[1]="te"
d[2]="hi"
d[3]="en"
d[4]="ta"
d[5]="ml"

n=Translator()

if selection=="tts":
   
    txt=pytesseract.image_to_string(img)
    if len(txt)==0:
        txt="the given image does not contains text or image is not clear"
    print(txt)
    
    text_translate=n.translate(txt,dest=d[a])
    text_translate=text_translate.text
    tts=gTTS(text=text_translate,lang=d[a],slow=False)
    tts.save('hell.mp3')
    playsound('hell.mp3')
   
else:
    try:
        frame=img
        l=[]
        result=model(frame)
        df=result.pandas().xyxy[0]
        for ind in df.index:
            x1,y1= int(df['xmin'][ind]), int(df['ymin'][ind])
            x2,y2= int(df['xmax'][ind]), int(df['ymax'][ind])

            label=df['name'][ind]

            cv2.rectangle(frame,(x1,y1),(x2,y2) ,(255,255,0),2)
            cv2.putText(frame,label,(x1,y1-5),cv2.FONT_HERSHEY_PLAIN,2,(255,255,0),2)
            l.append(label)
        if "person" in l:
            
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
        
        k=0
        if l.count("person")<len(face_names):
            l=[j for j in l if j!="person"]
            l.extend(face_names)
        elif "person" in l and len(face_names)==0:
            for j in range(len(l)):
                if l[j]=="person":
                    l[j]="unknown person"
        else:
            for j in range(len(l)):
                if l[j]=="person":
                    l[j]=face_names[k]
                    k+=1
        
        txt=""
        for i in l:
            txt=txt+i+" "

        text_translate=n.translate(txt,dest=d[a])
        text_translate=text_translate.text
        tts=gTTS(text=text_translate,lang=d[a],slow=False)
        ti="ab.mp3"
        tts.save(ti)

        cv2.imshow("Image", img)
        playsound(ti)
        
    except:
        txt="picture is not clear due to dim light"
        text_translate=n.translate(txt,dest=d[a])
        text_translate=text_translate.text
        tts=gTTS(text=text_translate,lang='en',slow=False)
        ti="ab1.mp3"
        tts.save(ti)
        playsound(ti)
    



