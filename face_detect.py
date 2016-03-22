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
    	frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel) # Blurs Face
    	cv2.circle(frame, (x+w/2, y+h/2), (w/2), (40, 40, 40), (5)) # Cirlce around face
    	cv2.circle(frame, (x+w/2 - 55, y+h/2 - 23), (6), (0, 0, 0), (5)) # Left eye
    	cv2.circle(frame, (x+w/2 + 55, y+h/2 - 23), (6), (0, 0, 0), (5)) # Right eye
    	cv2.ellipse(frame, (x+w/2, y+h/2 + 20), (70, 50), 20, 90, 30, (0,0,150), (5)) # Smile
#    	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255)) # Was used to draw rectangle around face

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()