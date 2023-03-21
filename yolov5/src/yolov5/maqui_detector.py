#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

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
        self.sub = rospy.Subscriber("/maqui/camera/front/image_raw", Image, self.callback)
        self.pub = rospy.Publisher("/maqui/camera/front/image_detected", Image, queue_size=10)
        self.pubBoxes = rospy.Publisher("/camera/color/detected_data", BoundingBoxes, queue_size=10)

        self.model = YoloV5(weights='/home/ignacio/uchile_ws/ros/maqui/soft_ws/src/uchile_vision/yolov5/yolov5_jp.pt') # poner aqui el path completo de los pesos
        self.names = self.model.names

    def callback(self, msg):
        self.img = self.br.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        detections = self.model.detect(self.img)

        # creamos el mensaje que será la lista de bounding boxes
        boxes = BoundingBoxes()
        if not detections == []:
            for i, (x1, y1, x2, y2, cls_conf, i_label) in enumerate(detections):
                _x = int(x1.item()) #round(x2.item() - x1.item())/2 + x1.item()
                _y = int(y1.item())
                _w = int(round(x2.item() - x1.item()))
                _h = int(round(y2.item() - y1.item()))
                label = str(self.names[int(i_label.item())])
                prob = cls_conf.item()
                # creando el mensaje
                box = BoundingBox()
                box.probability = prob
                box.xmin = x1
                box.ymin = y1
                box.xmax = x2
                box.ymax = y2
                box.id = i_label.item()
                box.Class = label
                #la ponemos en la lista de bounding boxes
                boxes.bounding_boxes.append(box)
                
                self.img = cv2.rectangle(self.img, (_x, _y), (_x +_w, _y + _h), (0,255,0), 2)
                self.img = cv2.putText(self.img, label, (_x,_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                print("detecte un",label, " con confianza de ", cls_conf.item())
        self.pubBoxes.publish(boxes)
        self.pub.publish(self.br.cv2_to_imgmsg(self.img, "bgr8"))

def main():
    Det = detector()
    rospy.init_node("yolov5", anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Apagando modulo de deteccion de imagenes")

if __name__=="__main__":
    main()