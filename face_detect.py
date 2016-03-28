""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
kernel = np.ones((21,21),'uint8')
radius = 10
black = (0,0,0)
white = (255,255,255)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))

    for (x,y,w,h) in faces:
    	frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:],kernel)
    	radius = int(h/15)      # eye radius
    	radius2 = int(.6*radius)# pupil/smile radius
    	xpos1 = x+int(.33*w)    # left x position
    	xpos2 = x+int(.66*w)    # right x position
    	ypos = y+int(.4*h)		# height for eyes

    	# drawing the eyes
    	cv2.circle(frame, (xpos1,ypos), radius, white, thickness = -1)
    	cv2.circle(frame, (xpos2,ypos), radius, white, thickness = -1)

    	# drawing the pupils
    	cv2.circle(frame, (xpos1,ypos), radius2, black, thickness = -1)
    	cv2.circle(frame, (xpos2,ypos), radius2, black, thickness = -1)

    	# drawing the smile
    	cv2.ellipse(frame, (x+int(.5*w),y+int(.65*h)), (int(w/4), int(h/5)), 0, 0, 180, black, radius2)
    	#cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
    	
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()