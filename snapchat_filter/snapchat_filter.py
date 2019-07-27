import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('Train/third-party/data/haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier("Train/third-party/frontalEyes35x16.xml")
nose_cascade = cv2.CascadeClassifier("Train/third-party/Nose18x15.xml")

img = cv2.imread("Test/Before.png",-1)
# glasses = cv2.imread("Train/glasses.png",-1)
# mustache = cv2.imread("Train/mustache.png",-1)

mustache = cv2.imread("Train/glasses.png",-1)
glasses = cv2.imread("Train/mustache.png",-1)

cv2.imshow('Original',img)

## For glasses

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eyes = eyes_cascade.detectMultiScale(gray, 1.3, 5)

i = 0
for (x, y, w, h) in eyes:
    cv2.rectangle(gray, (x, y), (x + w, y + h), (125, 155, 155), 3)
    i = 1
    break

if i == 0:
    print("not found")
    exit()

img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
glasses = cv2.resize(glasses, (w, h))

w, h, c = glasses.shape

for i in range(0, w):
    for j in range(0, h):

        if glasses[i, j][3] != 0:
            img[y + i, x + j] = glasses[i, j]

## For Mustache
mouth = nose_cascade.detectMultiScale(gray, 1.3, 5)

i = 0
for (x, y, w, h) in mouth:
    cv2.rectangle(gray, (x, y), (x + w, y + h), (125, 255, 125), 3)
    i = 1

if i == 0:
    print("not found")
    exit()

mustache = cv2.resize(mustache,(w,h))
w,h,c = mustache.shape

y+=int(h)
x+=int(w/2)

for i in range(0,w):
    for j in range(0,h):

        if mustache[i,j][3] !=0:
            img[y+i,x+j] = mustache[i,j]



cv2.imshow('snapchat_filter', img)

cv2.waitKey(0)