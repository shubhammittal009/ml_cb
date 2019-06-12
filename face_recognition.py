import cv2
import numpy as np

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

while True:
	ret,frame = cam.read()
	if ret == False:
		print("Something Went Wrong!")
		continue

	key_pressed = cv2.waitKey(1) & 0xFF #Bitmasking to get last

	if key_pressed == ord('q'): #ord--> ASCII Values(8 bit)
		break

	faces = face_cascade.detectMultiScale(frame,1.3,5)
	# print(faces)

	if(len(faces)==0):
		cv2.imshow("video",frame)
		continue
	for face in faces:
		x,y,w,h = face
		face_section = frame[y-10:y+h+10,x-10:x+w+10]
		face_section = cv2.resize(face_section,(100,100))
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	cv2.imshow("video",frame)                       
	cv2.imshow("frame",face_section)
	# cv2.imshow("GRAYSCALE Title",gray)

cam.release()
cv2.destroyAllWindows()