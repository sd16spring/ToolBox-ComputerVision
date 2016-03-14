import numpy as np
import cv2

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('/home/jkunimune/ToolBox-ComputerVision/input.mp3')
kernel = np.ones((21,21),'uint8')
face_cascade = cv2.CascadeClassifier('/home/jkunimune/ToolBox-ComputerVision/haarcascade_frontalface_alt.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)   # makes image face the right way
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))
    for (x,y,w,h) in faces:
        frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel)
        cv2.ellipse(frame, (x+w/2,y+h/2), (w/2,h/2), 0,0,360, (0,255,255), thickness=-1)
        cv2.ellipse(frame, (x+w/2,y+h/2), (w/2,h/2), 0,0,360, (0,0,0), thickness=3)
        cv2.line(frame, (x+w/3,y+h/4), (x+w/3,y+h/2), (0,0,0), thickness=3)
        cv2.line(frame, (x+2*w/3,y+h/4), (x+2*w/3,y+h/2), (0,0,0), thickness=3)
        cv2.ellipse(frame, (x+w/2,y+h/2), (w/3,h/3), 0,0,180, (0,0,0), thickness=3)
     # Display the resulting frame
    #cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()