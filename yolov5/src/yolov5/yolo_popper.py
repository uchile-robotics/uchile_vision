#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('cv_bridge')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from detect import *
import torch
import torchvision.transforms as transforms

class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("yolo_detection",Image)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/maqui/camera/front/image_raw",Image,self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    # transform = transforms.ToTensor()
    # tensor = transform(cv_image)
    detections = YoloV5.detect(cv_image)     
   
    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(detections, "bgr8"))
    except CvBridgeError as e:
      print(e)

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)