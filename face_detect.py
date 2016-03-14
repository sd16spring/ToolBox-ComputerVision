"""
Author: Cedric Kim
Experiment with face detection and image filtering using OpenCV """


import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/home/cedric/ToolBoxes/ToolBox-ComputerVision/haarcascade_frontalface_alt.xml')
kernel = np.ones((50,50),'uint8')

while(True):

    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))
    for (x,y,w,h) in faces:
        frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel) ##blurs image
        cv2.circle(frame, (x+w/2, y+h/2), (w/2), (0, 255, 0), (5))                  ##create head
        cv2.circle(frame, (x+w/2 - 30, y+h/2 - 50), (6), (0, 0, 0), (5))            ##create eyes
        cv2.circle(frame, (x+w/2 + 30, y+h/2 - 30), (6), (0, 0, 0), (5))
        cv2.ellipse(frame, (x+w/2, y+h/2), (70, 50), 20, 90, 30, (0,0,0), (5))      ##create smile
        #cv2.ellipse(frame,(256,256),(100,50),0,0,180,255,-1)
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255)) ##draws the rectangle
     # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()