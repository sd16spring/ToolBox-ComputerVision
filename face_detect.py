""" Experiment with face detection and image filtering using OpenCV """

import numpy as np
import cv2

capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/home/joseph/ToolBox-ComputerVision/haarcascade_frontalface_alt.xml')
kernel = np.ones((21,21),'uint8')
while(True):
	# Capture frame-by-frame
	ret, frame = capture.read()
	faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))
	for (x,y,w,h) in faces:
		frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel)#  Blurs the face using dilate command
		cv2.circle(frame, (x+int(.333*w),y+int(.333*h)), int(.125*h), (255,255,255), -1)
		cv2.circle(frame, (x+int(.333*w),y+int(.333*h)), int(.0625*h), (5,5,5), -1)
		cv2.circle(frame, (x+int(.666*w),y+int(.333*h)), int(.125*h), (255,255,255), -1)
		cv2.circle(frame, (x+int(.666*w),y+int(.333*h)), int(.0625*h), (5,5,5), -1)
		cv2.line(frame, (x+int(.35*w),y+int(.75*h)), (x+int(.7*w),y+int(.75*h)), (60,60,200),5)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0))

	# Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



capture.release()
cv2.destroyAllWindows()

