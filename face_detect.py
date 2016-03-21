""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
# this is for recognizing faces and putting a box around them
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# this is for blurring inside the aforementioned box
kernel = np.ones((21, 21), 'uint8')

while(True):
    # capture frame-by-frame
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20, 20))
    for (x, y, w, h) in faces:
        frame[y:y+h, x:x+w, :] = cv2.dilate(frame[y:y+h, x:x+w, :], kernel)
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))
        # draw the mouth
        cv2.ellipse(frame, (int(x+w/2), int(y+h/1.4)), (int(w/5), int(y/6)), 0, 0, 180, 0, 10, 1)
        # draw the scleras
        cv2.circle(frame, (int(x+w/4), int(y+h/3)), w/10, (255,255,255), -1)
        cv2.circle(frame, (int(x+3*w/4), int(y+h/3)), w/10, (255,255,255), -1)
        # draw the pupils
        cv2.circle(frame, (int(x+w/4), int(y+h/3)), w/30, 0, -1)
        cv2.circle(frame, (int(x+3*w/4), int(y+h/3)), w/30, 0, -1)
    # display resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release capture when everything done
cap.release()
cv2.destroyAllWindows()
