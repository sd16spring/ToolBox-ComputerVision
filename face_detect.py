""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('/home/becca/Documents/Software Design/ToolBox-ComputerVision/haarcascade_frontalface_alt.xml')
kernel = np.ones((40,40),'uint8')

cap = cv2.VideoCapture(0)

while(True):
    
    # caprute video and detect face
	ret, frame = cap.read()
	faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))
	
	# blur out face
	for (x,y,w,h) in faces:
		frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel)
		#cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))

	# overlay funny face inside rectangle	
	for (x,y,w,h) in faces:
		cv2.circle(frame,(int(x+w/2.0),int(y+h/2.0)),112,(255,255,255)) # circle around face 
		cv2.circle(frame,(int(x+w/3.5),int(y+h/2.5)),25,(255,255,255), -1) # left eyeball
		cv2.circle(frame,(int(x+w*2.6/3.5),int(y+h/2.5)),40,(255,255,255),-1) # right eyeball
		cv2.circle(frame,(int(x+w/3.5),int(y+h/2.5)+2),8,(204,0,204),-1) # left pupil
		cv2.circle(frame,(int(x+w*2.6/3.5),int(y+h/2.5)+2),12,(204,0,204),-1) # right epupil
		cv2.ellipse(frame,(int(x+w/2.0),int(y+h*3/4)),(40,10),10,20,180,(0,0,0),10) # smile


	# Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()