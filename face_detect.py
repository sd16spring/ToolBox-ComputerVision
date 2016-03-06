""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',0, 20.0, (640,480))
face_cascade = cv2.CascadeClassifier('/home/xiaozheng/Softdes/ToolBox-ComputerVision/haarcascade_frontalface_alt.xml')
kernel = np.ones((30,30),'uint8')
while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# out.write(frame)

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))

	for (x,y,w,h) in faces:
		frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:],kernel)
		# cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
		cv2.circle(frame,(int(x+w/3.5),int(y+h/2.5)),20,(255,255,255),-1) #left eyewhite
		cv2.circle(frame,(int(x+w*2.6/3.5),int(y+h/2.5)),20,(255,255,255),-1) #right eyewhite
		cv2.circle(frame,(int(x+w/3.5),int(y+h/2.5)+2),8,(0,0,0),-1) #left eyeball
		cv2.circle(frame,(int(x+w*2.6/3.5),int(y+h/2.5)+2),8,(0,0,0),-1) #right eyeball
		# cv2.ellipse(frame,(int(x+w/2.0),int(y+h*3/4)),(40,10),0,0,180,(0,0,0),10) #smiley face
		cv2.ellipse(frame,(int(x+w/2.0),int(y+h*3/4)),(40,10),0,0,-180,(0,0,0),10) #frowney face
	# Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()