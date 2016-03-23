""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
kernel = np.ones((21,21),'uint8')
face_cascade = cv2.CascadeClassifier('/home/guokevin/ToolBox-ComputerVision/frontal.xml')
# print face_cascade

while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))
    for (x,y,w,h) in faces:
        cv2.ellipse(frame, (x+w/2,y+h/2), (3*w/7,2*h/3), 0,0,180,(0,0,0), thickness=-1)
        cv2.ellipse(frame, (x+w/2,y+h/3), (7*w/14,10*h/12), 0,0,-180,(0,0,0), thickness=-1)
        points = np.array([[(x+w/15), y+h/3],[x, y+h/2.5], [x-w/4, y+h/5], [x-w/4, y+h/4], [x-w/2, y+h/5]])
        cv2.polylines(frame,np.int32([points]), False, (0,0,0),w/16)
        left = np.array([[x+w/20, y+h/3],[x+w/8,y+h/1.6]])
        cv2.polylines(frame,np.int32([left]), False, (0,0,0),w/13)
        right = np.array([[x+w-w/20, y+h/3],[x+w-w/8,y+h/1.6]])
        cv2.polylines(frame,np.int32([right]), False, (0,0,0),w/13)
    #Display the resulting frame
    cv2.imshow('frame',frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()