import cv2
import streamlit as st

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_BUFFERSIZE, 2)
webcam.set(cv2.CAP_PROP_FPS, 60)
stframe=st.empty()
sttext=st.empty()
while True:
	successful_frame_read, frame=webcam.read()
	
	grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
	face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
	frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	for (x,y,w,h) in face_coordinates:
		cv2.rectangle(frame,(x,y),(x+w,y+h) ,(0,0,256, 5))
	frame=cv2.flip(frame,1)
	
	stframe.image(frame,channels='GRAY',output_format="JPEG")
	sttext.write("happy")

