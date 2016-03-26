""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
kernel = np.ones((100,100),'uint8')

while(True):
	ret, frame = cap.read()
	faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))
	for (x,y,w,h) in faces:
		frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel)
		cv2.circle(frame, (x+w/3,y+h/3), 25, [255, 255, 255], -10)
		cv2.circle(frame, (x+w*2/3,y+h/3), 25, [255, 255, 255], -10)
		cv2.circle(frame, (x+w/3,y+h/3), 10, [0,0,0], -10)
		cv2.circle(frame, (x+w*2/3,y+h/3), 10, [0,0,0], -10)
		cv2.ellipse(frame,(x+w/2,y+h*2/3),(50,30),0,0,180,0,10)
		#cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))q
	 # Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()