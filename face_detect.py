""" Experiment with face detection and image filtering using OpenCV """

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

""" initialize the face detector program """
face_cascade = cv2.CascadeClassifier('/home/harper/ToolBox-ComputerVision/haarcascade_frontalface_alt.xml')

""" create a numpy matrix that controlls the degree of blurring 
(larger matrix = more bluring)"""
kernel = np.ones((21,21),'uint8')

while(True):
    # Capture frame-by-frame
    
	ret, frame = cap.read()
	faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))
	
	# this loop creates a rectangle over the detected face area
	# then blurs it with the kernel
	for (x,y,w,h) in faces:
	    frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel)
	    cv2.circle(frame, ((x + w/3), (y + h/3)), 15, (0,0,0), thickness=-1)
	    cv2.circle(frame, ((x + 2*w/3), (y + h/3)), 15, (0,0,0), thickness=-1)
	    cv2.ellipse(frame, ((x + w/2), (y + 2*h/3)), (50, 25) , 180, 180, 360, (0,0,0), thickness=5)
	    cv2.ellipse(frame, ((x + w/2 - 10), (y + 2*h/3 + 30)), (15, 25) , 50, -30, 160, (0,0,255), thickness=5) 
	    #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))

	# Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()