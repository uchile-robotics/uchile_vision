#!/usr/bin/env python
# -*- coding: utf-8 -*-

import imutils
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import cv2

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

import cv2
import tf
import tf2_ros
import rospy
import ros_numpy
import numpy as np
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from darknet_ros_msgs.msg import BoundingBox, BoundingBoxes

from yolov5.detect import YoloV5

class detector:
    def __init__(self):

        self.br =  CvBridge()
        self.img = None
        # package usb camara + cambiar la ruta de abajo
        self.sub = rospy.Subscriber("/bender/sensors/rgbd_head/rgb/image_rect_color", Image, self.callback)
        self.pub = rospy.Publisher("/camera/color/image_detected", Image, queue_size=10)
        self.pubBoxes = rospy.Publisher("/camera/color/detected_data", BoundingBoxes, queue_size=10)


    def upper_detector(self, frame):
        orig = frame.copy()
        frame = imutils.resize(self.img, width=min(400, frame.shape[1]))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        # Detecting upperbody
        upperBody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')    
        upperBodydetections= upperBody_cascade.detectMultiScale(frame_gray, scaleFactor = 1.05)

        return upperBodydetections
    
    def fullbody_detector(self, frame):
        orig = frame.copy()
        frame = imutils.resize(self.img, width=min(400, frame.shape[1]))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        # Detecting fullbody
        (fullBodydetections, _) = hog.detectMultiScale(frame_gray, winStride=(5, 5), padding=(3, 3), scale = 1.05)

        return fullBodydetections

    def callback(self, msg):
        self.img = self.br.imgmsg_to_cv2(msg, desired_encoding = "bgr8")
        detections_full = self.fullbody_detector(self.img)
        detections_upper = self.upper_detector(self.img)

        # creamos el mensaje que será la lista de bounding boxes
        boxes = BoundingBoxes()
        if not detections_full == []:
            for i, (x1, y1, x2, y2) in enumerate(detections_full): #esto va a dar error
                _x = int(x1.item()) #round(x2.item() - x1.item())/2 + x1.item()
                _y = int(y1.item())
                _w = int(round(x2.item() - x1.item()))
                _h = int(round(y2.item() - y1.item()))
                label = 'fullbody'
                prob = 0.0

                # creando el mensaje
                box = BoundingBox()
                box.probability = prob
                box.xmin = x1
                box.ymin = y1
                box.xmax = x2
                box.ymax = y2
                box.id = 0
                box.Class = label
                
                #la ponemos en la lista de bounding boxes
                boxes.bounding_boxes.append(box)
                self.img = cv2.rectangle(self.img, (_x, _y), (_x +_w, _y + _h),(255,0,0), linewidht = 2)
                self.img = cv2.putText(self.img, label, (_x,_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1)
                print("detecte un " + str(label))
                
        if not detections_upper == []:
            for i, (x1, y1, x2, y2) in enumerate(detections_full): #esto va a dar error
                _x = int(x1.item()) #round(x2.item() - x1.item())/2 + x1.item()
                _y = int(y1.item())
                _w = int(round(x2.item() - x1.item()))
                _h = int(round(y2.item() - y1.item()))
                label = 'upperbody'
                prob = 0.0

                # creando el mensaje
                box = BoundingBox()
                box.probability = prob
                box.xmin = x1
                box.ymin = y1
                box.xmax = x2
                box.ymax = y2
                box.id = 0
                box.Class = label
                
                #la ponemos en la lista de bounding boxes
                boxes.bounding_boxes.append(box)
                self.img = cv2.rectangle(self.img, (_x, _y), (_x +_w, _y + _h), (0,0,255), linewidht = 2)
                self.img = cv2.putText(self.img, label, (_x,_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1)
                print("detecte un " + str(label))

        self.pubBoxes.publish(boxes)
        self.pub.publish(self.br.cv2_to_imgmsg(self.img, "bgr8"))

def main():
    Det = detector()
    rospy.init_node("hog", anonymous=True)
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Apagando modulo de deteccion de imagenes")

if __name__=="__main__":
    main()



