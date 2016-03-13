""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/home/nathan/softwareDesign/ToolBox-ComputerVision/haarcascade_frontalface_alt.xml')
kernel = np.ones((21,21),'uint8')

while(True):
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))

    for (x,y,w,h) in faces:
        frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel)
        cv2.rectangle(frame,(x+w/6,y+h/4),(x+w/6+w/4,y+h/4+h/4),(0,0,255))
        cv2.rectangle(frame,(x+w/6+w/2,y+h/4),(x+w/6+w/4+w/2,y+h/4+h/4),(0,0,255))
        cv2.circle(frame, (x+w/2, y+5*h/6) , w/8, (0,0,255), thickness=1, lineType=8, shift=0)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
