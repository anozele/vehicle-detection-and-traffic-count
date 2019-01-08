import cv2
import numpy as np


cam = cv2.VideoCapture('car.mp4')
color=(0,255,0)
width=2
radius=100
point=(0,0)
global blobs
carcascade = cv2.CascadeClassifier('cars.xml')
count=0
while True:
    ret, frame = cam.read()
    frame=cv2.resize(frame,point,fx=0.8,fy=0.9)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = carcascade.detectMultiScale(gray,scaleFactor=1.04,minNeighbors=8,minSize=(50,50))

    for (x,y,w,h) in cars:
        p=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    

    cv2.imshow('traffic countt', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
