import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import os 
import csv
import sys

face_cascade = cv2.CascadeClassifier('Train/third-party/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier("Train/third-party/frontalEyes35x16.xml")
nose_cascade = cv2.CascadeClassifier("Train/third-party/Nose18x15.xml")

img = cv2.imread("Test/Before.png",-1)
glasses = cv2.imread("Train/glasses.png",-1)
mustache = cv2.imread("Train/mustache.png",-1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    cv2.rectangle(img,(x,y),(x+w,y+h),(255, 255, 255), 3)
    
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0, 255, 0), 3)
        roi_eyes = gray[ey:ey+eh, ex:ex+ew]
        
    nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
    for (nx,ny,nw,nh) in nose:
        cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0, 255, 255), 2)
        roi_eyes = gray[ny:ny+nh, nx:nx+nw]

cv2.imshow('img', img)