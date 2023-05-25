# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

cv2.startWindowThread()
cap = cv2.VideoCapture(0)

hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 


while(True):
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=min(400, frame.shape[1]))
    orig = frame.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # Detecting upperbody
    upperBody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')    
    upperBodydetections= upperBody_cascade.detectMultiScale(frame_gray, scaleFactor = 1.05)
    for (x,y,w,h) in upperBodydetections:
        cv2.rectangle(orig,(x,y),(x+w,y+h), (255,0,0), 2)

    # Detecting fullbody
    (fullBodydetections, _) = hog.detectMultiScale(frame_gray, winStride=(5, 5), padding=(3, 3), scale = 1.05)

    for (x, y, w, h) in fullBodydetections: 
      cv2.rectangle(orig, (x, y),(x + w, y + h),(0, 0, 255), 2) 
      
    print('Human Detected : ', len(fullBodydetections))
    
    # Display the resulting frame
    cv2.imshow('frame', orig)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

