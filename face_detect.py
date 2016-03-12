""" Experiment with face detection and image filtering using OpenCV """
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/home/willem/Programming/ToolBox-ComputerVision/haarcascade_frontalface_alt.xml')
kernel = np.ones((21,21),'uint8')


while(True):
	ret, frame = cap.read()
	faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))

	for (x,y,w,h) in faces:
	    frame[y:y+h,x:x+w,:] = cv2.dilate(frame[y:y+h,x:x+w,:], kernel)
	    #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
	    #draw cartoon face
	    cv2.ellipse(frame, ((x+int(w/2)),(y+int(h/1.4))), ((int(w/4)),(int(h/4))), 180, 200, 340, (0,0,0), 12, 8, 0) 
	    cv2.circle(frame, ((x+int(w/3)),(y+int(h/3))), int(w/10), (0,0,0), 12, 8, 0)
	    cv2.circle(frame, ((x+int(w)-int(w/3)),(y+int(h/3))), int(w/10), (0,0,0), 12, 8, 0)

	    #angry brow
	    cv2.line(frame, (x+int(w/7),y+int(h/10)), ((x+int(w/2.2)),(y+int(h/5))), (0,0,0), 12, 8, 0) 
	    cv2.line(frame, (x+w -int(w/7),y+int(h/10)), ((x+w -int(w/2.2)),(y+int(h/5))), (0,0,0), 12, 8, 0) 

	 # Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()