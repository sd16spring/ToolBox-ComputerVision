""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml')
kernel = np.ones((21,21),'uint8')
cap = cv2.VideoCapture(0)

def draw_face_smile():
	a = cv2.ellipse2Poly((int(x+w/3), int(y+h/2.3)), (w/10,h/8), 0, 0, 360, 10)
	b = cv2.ellipse2Poly((int(x+w-w/3), int(y+h/2.3)), (w/10,h/8), 0, 0, 360, 10)
	cv2.fillConvexPoly(frame, a, (255,255,255))
	cv2.fillConvexPoly(frame, b, (255,255,255))
	cv2.ellipse(frame,(int(x+w/3), int(y+h/2.3)), (w/10,h/8), 0, 0, 360,(45,82,160), 5)
	cv2.ellipse(frame,(int(x+w-w/3), int(y+h/2.3)), (w/10,h/8), 0, 0, 360,(45,82,160), 5)
	cv2.circle(frame, (int(x+w/3), int(y+h/2.3+h/15)), 20,(10,49,109), -1)
	cv2.circle(frame, (int(x+w-w/3), int(y+h/2.3+h/15)), 20,(10,49,109), -1)
	cv2.ellipse(frame,(int(x+w/2), int(y-h/6)), (int(w/2),h), 0, 60, 120, (45,82,160),5)
	cv2.circle(frame, (int(x+w/2), int(y+h/2)), w/2,(45,82,160), 5)


while(True):
	ret, frame = cap.read()
	faces = face_cascade.detectMultiScale(frame, scaleFactor = 1.2, minSize = (20,20))
	for (x,y,w,h) in faces:
		frame[y:y+h, x:x+w, :] = cv2.dilate(frame[y:y+h,x:x+w,:],kernel)
		#cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))
		draw_face_smile()
	cv2.imshow('frame', frame)


	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()