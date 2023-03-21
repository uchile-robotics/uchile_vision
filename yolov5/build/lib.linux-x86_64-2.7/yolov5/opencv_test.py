#!/usr/bin/env python

from __future__ import print_function

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox


class distance_to_object:

  def __init__(self):
    self.image_pub = rospy.Publisher("img_processed",Image,queue_size=5)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.img_callback)
    self.depth_img_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
    self.bb_sub = rospy.Subscriber("/camera/color/detected_data", BoundingBoxes, self.bb_callback)
    self.img = None
    self.depth_img = None
    self.output = None
    self.d = None
    self.depth_raw = None


  def bb_callback(self,data): 
    thickness = 2
    color = (255, 0, 0)
    image = self.depth_img
    out_img = self.img
    i_max = 0
    p_max = 0   

    for i in range(len(data.bounding_boxes)):
        box = data.bounding_boxes[i]
        p,xmin,xmax,ymin,ymax = box.probability, box.xmin, box.xmax, box.ymin, box.ymax

        if p > p_max:
            p_max = p   
            start_point = (xmin, ymin)
            end_point = (xmax, ymax)
    print("xmin: ", xmin)
    print("xmax: ", xmax)
    self.d = self.depth_raw[(ymax+ymin)//2, (xmax+xmin)//2]/10   #IndexError: index 497 is out of bounds for axis 0 with size 480
    print("distancia al obj es ",self.d," centimetros\n")
    out_img = cv2.rectangle(out_img, start_point, end_point, color, thickness)
    txt = "distancia = " + str(self.d) + " cm"
    self.output = cv2.putText(out_img,txt,(ymin,xmin-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2)




  def img_callback(self,data):
    try:
        cv_image = self.bridge.imgmsg_to_cv2(data,"bgr8")
        self.img = cv_image
    except CvBridgeError as e:
      print(e)

    (rows,cols,channels) = cv_image.shape


  def depth_callback(self,data):
    try:
        data.encoding='mono16'
        cv_image_d = self.bridge.imgmsg_to_cv2(data, "mono16") 
    except CvBridgeError as e:
      print(e)

    (rows,cols) = cv_image_d.shape
    #self.depth_img = cv_image_d

    img_n = cv2.normalize(src=cv_image_d, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    im_color = cv2.cvtColor(img_n, cv2.COLOR_GRAY2BGR)
    self.depth_img = im_color
    self.depth_raw = cv_image_d
    if self.img is not None and self.output is not None:
        #concat = np.concatenate((self.img, im_color), axis=1)
        try:
            concat = np.concatenate((self.img,self.output), axis=1)
            cv2.imshow("Imagenes", concat)
            cv2.waitKey(3)
        except:
            print("tiro error")

    #try:
      #self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image_d, "mono16"))
    #except CvBridgeError as e:
      #print(e)

def main(args):
  ic = distance_to_object()
  rospy.init_node('distance_to_object', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)