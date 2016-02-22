""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/home/emily/SoftDes/ToolBox_Exercises/ToolBox-ComputerVision/haarcascade_frontalface_alt.xml')
kernel = np.ones((40,40),'uint8')

while(True):
    ret, frame = cap.read()

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.5, minSize=(20,20))

    for (x,y,w,h) in faces:
    	#all calculations are here to make the code a little more readable
    	length = x + w
    	height = y + h
    	center_x = int(x + 0.5 * w)
    	center_y = int(y + 0.5 * h)
    	left_eye_x = int(x + 0.25 * w)
    	both_eyes_y = int(y + 0.25 * h)
    	right_eye_x = int(x + 0.75 * w)
    	big_circle_radius = int(0.5 * w)
    	smaller_circle_radius = int(w * 0.1)
    	pupil_radius = int(w * 0.05)
    	smile_y = int(y + 0.55 * h)
    	ellipse_x_axis = int(0.4 * w)
    	ellipse_y_axis = int(0.2 * h)


    	frame[y:height, x:length,:] = cv2.dilate(frame[y:height,x:length,:], kernel)
    	cv2.rectangle(frame,(x,y),(length,height),(0,0,255))
    	cv2.circle(frame,(center_x, center_y), big_circle_radius, (0,0,0), 3) #face outline
    	cv2.circle(frame,(center_x, center_y), big_circle_radius, (200,200,0), -1) #face fill
    	cv2.circle(frame,(left_eye_x, both_eyes_y), smaller_circle_radius, (1000,1000,1000), -1) #left eye, white
    	cv2.circle(frame,(right_eye_x, both_eyes_y), smaller_circle_radius, (1000,1000,1000), -1) #right eye, white
    	cv2.circle(frame,(left_eye_x+7, both_eyes_y+3), pupil_radius, (0,0,0), -1) #left eye, pupil
    	cv2.circle(frame,(right_eye_x-7, both_eyes_y+3), pupil_radius, (0,0,0), -1) #right eye, pupil
    	cv2.ellipse(frame,(center_x, smile_y), (ellipse_x_axis,ellipse_y_axis), 0,0,180, (0,0,0), 3) #smile!!

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()