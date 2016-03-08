""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np
from math import pi

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3,minSize=(20,20))
    for (x,y,w,h) in faces:
    	frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,], np.ones((w/5,h/5),'uint8'),iterations = 1)
        cv2.ellipse(frame, (w/2+x,h/2+y), (w/2,h/2),90, 0, 360, frame[w/2+x,h/3+y].tolist(),-1)
        cv2.circle(frame, (w/4+x,2*h/5+y), w/15,[0,0,0],-1)
        cv2.circle(frame, (3*w/4+x,2*h/5+y), w/15,[0,0,0],-1)
        cv2.ellipse(frame, (w/2+x,2*h/3+y), (w/5,h/5),90, -90, 90, [0,0,255],-1)
    	#cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()