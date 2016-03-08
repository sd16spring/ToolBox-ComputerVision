""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np


cap = cv2.VideoCapture('video.mp4')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
kernel = np.ones((40, 40), 'uint8')
frame_rate = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))


    for (x,y,w,h) in faces:
        # Blurs out face
        frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel)

        # Draws Eyes
        cv2.circle(frame, (x+(w/3), y+(h/3)), w/10, (255, 255, 255), -1)
        cv2.circle(frame, (x+(2*w/3), y+(h/3)), w/10, (255, 255, 255), -1)
        cv2.circle(frame, (x+(w/3), y+(h/3)), w/20, (0, 0, 0), -1)
        cv2.circle(frame, (x+(2*w/3), y+(h/3)), w/20, (0, 0, 0), -1)

        # Draws Mouth
        cv2.ellipse(frame, (x+(w/2), y+(3*h/5)), (w/3, h/5), 0, 180, 0, 0, h/20)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()