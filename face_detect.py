""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
kernel = np.ones((10,10),'uint8')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))
    for (x,y,w,h) in faces:
        frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel)
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
        center = ( (x+(w/2)), (y+(h/2)) )
        cv2.circle(frame, center, h/2, (0,125,255), 5)
        cv2.circle(frame, (center[0]-(w/6), center[1]-(h/7)), h/8, (0,255,255), 15)
        cv2.circle(frame, (center[0]+(w/6), center[1]-(h/7)), h/8, (0,255,255), 15)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()