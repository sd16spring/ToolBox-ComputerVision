""" Experiment with face detection and image filtering using OpenCV. This is
mostly copied code, though I tried to look through the docs and understand
what is going on. I made a rather silly annoyed face for the rectangle."""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    kernel = np.ones((21,21),'uint8')

    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
        for (x,y,w,h) in faces:
            frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
            cv2.circle(frame,(x+w/4,y+h*4/10),(w/16),(0,0,0),-1)
            cv2.circle(frame,(x+w*3/4,y+h*4/10),(w/16),(0,0,0),-1)
            cv2.line(frame,(x+w/4,y+h/5),(x+w/2,y+h*4/10),(0,0,0),3)
            cv2.line(frame,(x+w*3/4,y+h/5),(x+w/2,y+h*4/10),(0,0,0),3)
            cv2.line(frame,(x+w/4,y+h*3/4),(x+w*3/4,y+h*3/4),(0,0,0),3)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
