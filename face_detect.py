""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    face_cascade = cv2.CascadeClassifier('/home/anne/ToolBox-ComputerVision/haarcascade_frontalface_alt.xml')
    kernel = np.ones((21,21),'uint8')


    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))
    for (x,y,w,h) in faces:
        # print x
        # print y
        # print w
        # print h
#        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
#        for (x,y,w,h) in faces:
        frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
        #cv2.circle(frame, (x + w/2, y + h/2), int(x/2), (0,0,255))

        #cv2.ellipse(frame, (x + w/2, y + .75*h), (20, 10), 0, 0, 180, (0,0,255), -1)
        cv2.ellipse(frame,(x + w/2, int(y + .75*h)),(50,25),0,0,180,(0, 0, 0), thickness=5)
        #white Eyes:
        cv2.circle(frame, (int(x + .3*w), int(y + .4*h)), 20, (255, 255, 255), -1)
        cv2.circle(frame, (int(x + w - .3*w), int(y +.4*h)), 20, (255, 255, 255), -1)
        #pupils:
        cv2.circle(frame, (int(x + .3*w + 5), int(y + .4*h)), 5, (0, 0, 0), -1)
        cv2.circle(frame, (int(x + w -.3*w - 5), int(y + .4*h)), 5, (0, 0, 0), -1)


    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit() #break?

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

#Make a OpenCV Object
#Combining the While loops together