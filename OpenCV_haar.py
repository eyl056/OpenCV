
# coding: utf-8

# In[8]:

import cv2
import numpy as np
print 'Importing Libraries'


# In[7]:

import cv2

faceCascade = cv2.CascadeClassifier('/Users/eunyoung/Downloads/haarcascade_fullbody.xml')
#/Users/eunyoung/Downloads/
img_source = "pedest.jpg"

img = cv2.imread(img_source)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
)

for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+
w, y+h), (0, 255, 0), 2)

cv2.imshow('IMAGE',img)
cv2.waitKey()
cv2.destroyAllWindows()


# In[1]:

import cv2
from matplotlib import pyplot as plt 

#웹캠에서 영상을 읽어온다
cap = cv2.VideoCapture(0)
cap.set(3, 640) #WIDTH
cap.set(4, 480) #HEIGHT

#얼굴 인식 캐스케이드 파일 읽는다
face_cascade = cv2.CascadeClassifier('/Users/eunyoung/Downloads/haarcascade_frontalface_default.xml')

while(True):
    # frame 별로 capture 한다
    ret, frame = cap.read()
    
    if ret==True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.imread(frame,0)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        #인식된 얼굴 갯수를 출력
        print(len(faces))

        # 인식된 얼굴에 사각형을 출력한다
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        #화면에 출력한다
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print 'Importing Libraries'
    
cap.release()
cv2.destroyAllWindows()


# In[2]:

import cv2

#웹캠에서 영상을 읽어온다
cap = cv2.VideoCapture(0)
cap.set(3, 640) #WIDTH
cap.set(4, 480) #HEIGHT

#얼굴 인식 캐스케이드 파일 읽는다
face_cascade = cv2.CascadeClassifier('/Users/eunyoung/Downloads/haarcascade_frontalface_default.xml')

while(True):
    # frame 별로 capture 한다
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.imread(frame,0)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #인식된 얼굴 갯수를 출력
    print(len(faces))

    # 인식된 얼굴에 사각형을 출력한다
    for (x,y,w,h) in faces:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    #화면에 출력한다
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


# In[ ]:



