""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/home/linnea/ToolBox-ComputerVision/haarcascade_frontalface_alt.xml')
kernel = np.ones((21,21),'uint8')

while(True):
    # Capture frame-by-frame   

	ret, frame = cap.read()
	faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
		frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel)
	for (x,y,w,h) in faces:
		cv2.circle(frame, (x+90,y+120), 20, (0,0,255),10)
	for (x,y,w,h) in faces:
		cv2.circle(frame, (x+200,y+120), 20, (0,0,255),10)
	for (x,y,w,h) in faces:
		cv2.circle(frame, (x+150,y+240), 30, (50,50,255),25)
    	
    # Display the resulting frame
    	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()






