import cv2

cam = cv2.VideoCapture(0)

while True:
	ret,frame = cam.read()
	if ret == False:
		print("Something Went Wrong!")
		continue

	key_pressed = cv2.waitKey(1) & 0xFF #Bitmasking to get last

	if key_pressed == ord('q'): #ord--> ASCII Values(8 bit)
		break

	cv2.imshow("Video Title",frame)

cam.release()
cv2.destroyAllWindows()