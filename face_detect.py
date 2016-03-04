""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np

# Initiate face detector
face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml')
kernel = np.ones((40,40),'uint8')
## Get video from the webcam
cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))
	for (x,y,w,h) in faces:
		frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel)
		#cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
		# Draw the mouth
		cv2.ellipse(frame,(x+w/2, y+3*h/4),(w/4, h/6),0,0,180,(0,0,255),-1)
		# Draw the eyes
			# First the whites
		cv2.ellipse(frame,(x+w/4, y+h/3),(w/10, h/15),0,0,360,(255,255,255),-1)
		cv2.ellipse(frame,(x+3*w/4, y+h/3),(w/10, h/15),0,0,360,(255,255,255),-1)
			# Then the pupiles
		cv2.ellipse(frame,(x+w/4, int(y+(h/3)*1.05)),(w/20, h/20),0,0,360,(0,0,0),-1)
		cv2.ellipse(frame,(x+3*w/4, int(y+(h/3)*1.05)),(w/20, h/20),0,0,360,(0,0,0),-1)

	# Display the video
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

## Finish
cap.release()
cv2.destroyAllWindows()