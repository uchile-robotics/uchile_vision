#!/usr/bin/env python

from __future__ import print_function

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2 
from cv_bridge import CvBridge, CvBridgeError
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
from visualization_msgs.msg import Marker


class distance_to_object:

  def __init__(self):
    self.image_pub = rospy.Publisher("img_processed",Image,queue_size=5)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.img_callback)
    self.depth_img_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
    self.pcl2_sub = rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.pcl2_callback)
    self.bb_sub = rospy.Subscriber("/camera/color/detected_data", BoundingBoxes, self.bb_callback)
    self.marker_pub = rospy.Publisher("/camera/visualization_marker", Marker, queue_size = 2)
    self.pointcloud = None
    self.img = None
    self.depth_img = None
    self.output = None
    self.d = None
    self.depth_raw = None
    self.n = 0
    self.marker = None

  def pcl2_callback(self, msg):
     self.pointcloud = msg

  def bb_callback(self,data): 
    if self.n == 1:
       return
    
    thickness = 2
    color = (255, 0, 0)
    image = self.depth_img
    out_img = self.img
    out_pcl2 = self.pointcloud 
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

    row,col =  (ymax+ymin)//20, (xmax+xmin)//20
    if col > 497:
       col = 497 ## solo por ahora wea ordinaria
       
    if self.n == 0 and out_pcl2:
        point = pc2.read_points_list(out_pcl2, skip_nans=True, uvs = [(row,col)])[0] # point = (x,y,z,rgb)
        print('coordenadas de deteccion en pointcloud: ', point)
        print('distancia x pcl2: ', np.sqrt(point.x**2+point.y**2+point.z**2))

        marker = Marker()

        marker.header.frame_id = "camera_link"
        marker.header.stamp = rospy.Time.now()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = 2
        marker.id = 0

        # Set the scale of the marker
        marker.scale.x = 1.5
        marker.scale.y = 1.5
        marker.scale.z = 1.5

        # Set the color
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Set the pose of the marker
        marker.pose.position.x = point.y
        marker.pose.position.y = point.x
        marker.pose.position.z = point.z
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.lifetime.secs = 0
        self.marker = marker 
        self.marker_pub.publish(marker)

        

    self.d = self.depth_raw[row, col]   #IndexError: index 497 is out of bounds for axis 0 with size 480
    print("distancia al obj es ",self.d," centimetros\n")
    out_img = cv2.rectangle(out_img, start_point, end_point, color, thickness)
    txt = "distancia = " + str(self.d) + " cm"
    self.output = cv2.putText(out_img,txt,(ymin,xmin-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2)
    self.n = 1
    
  def img_callback(self,data):
    try:
        cv_image = self.bridge.imgmsg_to_cv2(data,"bgr8")
        self.img = cv_image
    except CvBridgeError as e:
      print(e)

    if self.marker is not None:
      self.marker_pub.publish(self.marker)

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