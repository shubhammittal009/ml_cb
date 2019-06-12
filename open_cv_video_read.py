import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
	ret,frame = cam.read()
	if ret == False:
		print("Something Went Wrong!")
		continue

	key_pressed = cv2.waitKey(1) & 0xFF #Bitmasking to get last

	if key_pressed == ord('q'): #ord--> ASCII Values(8 bit)
		break

	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	# print(frame.shape)
	# print(gray.shape)

	cv2.imshow("RGB Title",frame)
	cv2.imshow("GRAYSCALE Title",gray)

	cv2.imshow("RGB Title",frame)

cam.release()
cv2.destroyAllWindows()