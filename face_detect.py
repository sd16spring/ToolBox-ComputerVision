""" Experiment with face detection and image filtering using OpenCV """

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
kernel = np.ones((21,21),'uint8')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))
    for (x,y,w,h) in faces:
        frame[y:y+h, x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel)

    for (x,y,w,h) in faces:
        cv2.circle(frame, (x + int(w/2),y + int(h/2)), int(w/2), (0,255,255), thickness=-1)
        cv2.circle(frame, (x + int(w/4),y + int(h/4)), 10, (0,0,0), thickness=-1)

        cv2.circle(frame, (x + w - int(w/4),y + int(h/4)), 10, (0,0,0), thickness=-1)
        cv2.ellipse(frame, (x + int(w/2), y + h - int(h/4)), (int(w/4), int(h/5)), 180, 180, 360, (0,0,0))
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
