""" Experiment with face detection and image filtering using OpenCV """

"""
Completed by Kevin Zhang on 3/5/2016
Software Design Spring 2016
"""

import cv2
import numpy as np

#initalizing video of Donald Trump
cam = cv2.VideoCapture('Donald.mp4')

#initializing picture of chicken for defacing
chicken_raw = cv2.imread('chicken-head.png')

#captures the video and saves it as output, using a codec
fourcc = cv2.cv.CV_FOURCC(*'XVID') 
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (426,240), True)


while cam.isOpened():

	#face detector algorithm
	face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml')

	#blur thingy  matrix
	blur = np.ones((21,21), 'uint8')



	#capture frame by frame

	ret, frame = cam.read()

	if ret == True:


			

		faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20,20))

		for (x, y, w, h) in faces:
			frame[y:y+h,x:x+w] = cv2.dilate(frame[y:y+h,x:x+w], blur)

			#draws a chicken head on the face
			chicken = cv2.resize(chicken_raw, (w, h))
			frame[y:y+chicken.shape[0],x:x+chicken.shape[1]] = chicken


		out.write(frame)


		#draws a smiley face on my face
		# cv2.circle(frame, (x+w/3, y+h/3), 10, (0,255,0), -1, 8, 0)
		# cv2.circle(frame, (x+2*w/3, y+h/3), 10, (0,255,0), -1, 8, 0)
		# cv2.ellipse(frame, (x+w/2, y+h/2), (w/3, h/3), 0, 20, 160, (0,255,0), 10, 8, 0)

		#draws a rectangle around my face
		#cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0, 255))


		#display resulting frame 
		cv2.imshow('frame', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break		
	

cam.release()
out.release()
cv2.destroyAllWindows()

