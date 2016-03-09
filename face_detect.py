""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('cats.mp4')

# instantiate face and eye detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# numpy matrix controls degree of blurring (larger the matrix, more blurring)
kernel = np.ones((20,20), 'uint8')

while(True):
	# Capture video frame by frame
	ret, frame = cap.read()

	# get list of faces in the image
	faces = face_cascade.detectMultiScale(frame, scaleFactor = 1.2, minSize = (20,20))
	eyes = eye_cascade.detectMultiScale(frame)
	for (x,y,w,h) in faces:
		frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel)
		cv2.ellipse(frame, ((x+w)/2 + 60, y+h - (y+h)/5), (80, 60), 0, 180, 0, (255,255,255), -1)
		cv2.ellipse(frame, ((x+w)/2 + 60, y+h - (y+h)/5), (80, 60), 0, 180, 0, (0,0,0), 5)
		# cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(frame, (ex,ey), (ex+ew, ey+eh), (255,255,255), thickness = -1)
			cv2.ellipse(frame, (ex+ew-10,ey+eh), (10,30), 180, 180, 0, (0,0,0), -1)

	# Display resulting frame
	cv2.imshow('I Love Cats', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):				# 1 determines lag | 'q' determines escape key
		break

# Release capture
cap.release()
cv2.destroyAllWindows()