""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
temp = True
face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml')
kernel = np.ones((50,50),'uint8')
flag = 0
maxruntime = 50 #frames

while(temp == True):
	#Capture frame-by-frame
	ret, frame = cap.read()
	if not ret:
		print ret
		print frame
		temp == False


	#Operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))
	for (x,y,w,h) in faces:
		frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel)
		# Draws a red rectangle around your face.
		# cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))

		if flag > (maxruntime-5):
			# cv2.rectangle(frame,(x+150,y+50),(x+190,y+90),(0,0,255),-1)
			# cv2.rectangle(frame,(x+50,y+50),(x+90,y+90),(0,0,255),-1)
			cv2.ellipse(frame,(x+160,y+90),(30,40),30,180,360,(0,0,255),-1)
			cv2.ellipse(frame,(x+60,y+90),(30,40),150,0,180,(0,0,255),-1)
			cv2.line(frame,(x+40,y+90),(x+40,y+120),(0,0,255),8)
			cv2.line(frame,(x+70,y+60),(x+70,y+95),(0,0,255),8)
			cv2.line(frame,(x+160,y+60),(x+160,y+110),(0,0,255),8)
			if flag % 2 == 0:
				cv2.ellipse(frame,(x+120,y+175),(40,60),0,0,180,(0,0,0),-1)
			else:
				cv2.ellipse(frame,(x+120,y+175),(30,40),0,0,180,(0,0,0),-1)

		else:
			cv2.circle(frame,(x+200,y+150),20,(225,225,255),-1)
			cv2.circle(frame,(x+25,y+150),20,(225,225,255),-1)
			cv2.rectangle(frame,(x+150,y+50),(x+190,y+90),(0,0,0),-1)
			cv2.rectangle(frame,(x+50,y+50),(x+90,y+90),(0,0,0),-1)
			cv2.ellipse(frame,(x+120,y+175),(30,30),0,0,180,(0,0,0),5)

	#Display the resulting frame

	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	if flag > maxruntime:
		temp = cap.release() #if frame read correctly == True
		cv2.destroyAllWindows()

	flag = flag + 1

#release the capture
temp = cap.release() #if frame read correctly == True
cv2.destroyAllWindows()