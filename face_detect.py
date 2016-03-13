""" Anna Buchele
	This program takes your webcam footage, blurs out your face, replaces it with a smiley face, and then shows you the edited webcam footage. """

import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('/home/anna/ToolBox-ComputerVision/haarcascade_frontalface_alt.xml')

kernal = np.ones((40,40),'uint8')
cap = cv2.VideoCapture(0)
while(True):
	#frame by frame capture
	ret, frame = cap.read()

	"""assorted operations on frame:"""

	#turns frame grayscale
	#gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#flips frame upside down
	#frame=cv2.flip(gray,0)

	#changes size
	#ret = cap.set(3,320)
	#ret = cap.set(4,240)


	# finds faces
	faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))
	#draw a rectangle around face
	# for (x, y, w, h) in faces:
	#   cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
	#blurs out faces
	for (x,y,w,h) in faces:
		frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:],kernal)

		cv2.circle(frame, ((x+(w/4)),(y+(h/3))), w/12, (200,100,50), thickness=-1, lineType=8, shift=0) 
		cv2.circle(frame, ((x+(3*w/4)),(y+(h/3))), w/12, (200,100,50), thickness=-1, lineType=8, shift=0) 
		cv2.ellipse(frame, (x+w/2,y+3*h/4), (w/4,h/4), 180, 180, 360, (255,255,255), -1, 8, 0)


	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
#releasing the capture
cap.release()
cv2.destroyAllWindows()