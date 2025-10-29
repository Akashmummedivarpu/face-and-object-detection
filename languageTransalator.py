import pytesseract
from gtts import gTTS
from playsound import playsound
import os
#from tkinter import *
from googletrans import Translator
pytesseract.pytesseract.tesseract_cmd=r'C:/Program Files/Tesseract-OCR/tesseract.exe'
from PIL import Image
import cv2
img=cv2.imread("hhh.jpg")
#tts=gTTS(text="please select your language telugu press 1 for hindi press 2 for english press 3 for tamil press 4 for malyalam press 5 for kanada press 6",lang='en',slow=False)
#tts.save('hello.mp3')
#playsound('hello.mp3')

a=int(input())
#txt=pytesseract.image_to_string(img)
#print(txt)

txt="hello everyone"
n=Translator()
if a==2:
    text_translate=n.translate(txt,dest="hi")
    text_translate=text_translate.text
    tts=gTTS(text=text_translate,lang="hi",slow=False)
    tts.save('hell.mp3')
    playsound('hell.mp3')
elif a==1:
    text_translate=n.translate(txt,dest="te")
    text_translate=text_translate.text
    tts=gTTS(text=text_translate,lang="hi",slow=False)
    tts.save('hell.mp3')
    playsound('hell.mp3')
elif a==6:
    text_translate=n.translate(txt,dest="kn")
    text_translate=text_translate.text
    tts=gTTS(text=text_translate,lang="kn",slow=False)
    tts.save('hell.mp3')
    playsound('hell.mp3')
elif a==5:
    text_translate=n.translate(txt,dest="ml")
    text_translate=text_translate.text
    tts=gTTS(text=text_translate,lang="ml",slow=False)
    tts.save('hell.mp3')
    playsound('hell.mp3')
elif a==4:
    text_translate=n.translate(txt,dest="ta")
    text_translate=text_translate.text
    tts=gTTS(text=text_translate,lang="ta",slow=False)
    tts.save('hell.mp3')
    playsound('hell.mp3')
else:
    text_translate=n.translate(txt,dest="en")
    text_translate=text_translate.text
    tts=gTTS(text=text_translate,lang="en",slow=False)
    tts.save('hell.mp3')
    playsound('hell.mp3')
#os.system("start hello.mp3")
print(text_translate)  

