import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/home/youngdp/workspace/haarcascade_frontalface_alt.xml')
kernel = np.ones((21,21),'uint8')
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame',frame)
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))
    for (x,y,w,h) in faces:
    	print x, y
    	frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel)
    	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
    	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
    	cv2.circle(frame, (x + h/3, y + h / 3), 2, (0, 255, 0), 10, 8)
    	cv2.circle(frame, (x + 2* h/3, y + h / 3), 2, (0, 255, 0), 10, 8)
    	cv2.ellipse(frame, (x + w/2, y + h * 3 / 5), (w / 4, h /4), 150, 240, 360, (255, 0, 0), 5, 8)
    	 # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()