from unittest import result
from gtts import gTTS

import torch
import cv2
import tensorflow as tf  

cap=cv2.VideoCapture(0)

while True:
    
    _,frame=cap.read()
    model=torch.hub.load("WongKinYiu/yolov7","yolov7")

    '''img=cv2.imread("img6.jpg")
    img=cv2.resize(img,(800,650))'''

    result=model(frame)
    print(result)

    df=result.pandas().xyxy[0]
    print(df)

    for ind in df.index:

        x1,y1= int(df['xmin'][ind]), int(df['ymin'][ind])
        x2,y2= int(df['xmax'][ind]), int(df['ymax'][ind])

        label=df['name'][ind]

        cv2.rectangle(frame,(x1,y1),(x2,y2) ,(255,255,0),2)
        cv2.putText(frame,label,(x1,y1-5),cv2.FONT_HERSHEY_PLAIN,2,(255,255,0),2)
    cv2.imshow('IMAGE',frame)
    key=cv2.waitKey(1)
    if key & 0xFF==ord('q'):
        break
cap.release()
