#!/usr/bin/env python

import sys
import rospy
import cv2
import numpy as np

# ROS Imports
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs.point_cloud2 as pc2 

# Darknet 
from yolov5.detect import YoloV5

class RealSense_Marker_Generator():
    def __init__(self):

        rospy.init_node('RS_MG', anonymous=False)

        # ROS Pubs & Subs
        self.img_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.img_callback)
        self.depth_img_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.pointcloud_sub = rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.pointcloud_callback)
        self.marker_pub = rospy.Publisher("/camera/RS_MG/markers", MarkerArray, queue_size = 5)
        self.image_pub = rospy.Publisher("/camera/RS_MG/YOLO_detections",Image, queue_size=5)

        # Variables and object instances
        self.bridge = CvBridge()
        self.pointcloud = None
        self.img = None
        
        # YOLO module
        self.model = YoloV5(weights='/home/jaime/uchile_ws/ros/jaime/soft_ws/src/uchile_vision/yolov5/yolov5_jp.pt') # poner aqui el path completo de los pesos
        self.names = self.model.names
        self.colors = self.model.colors

        # Consider deleting:
        self.depth_img = None
        self.depth_raw = None

    def img_callback(self,msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg,"bgr8")
            self.img = cv_image
        except CvBridgeError as e:
            print(e)
    
    def depth_callback(self,data):
        return 
        # el resto queda por ahora de ref
        try:
            data.encoding='mono16'
            cv_image_d = self.bridge.imgmsg_to_cv2(data, "mono16") 
        except CvBridgeError as e:
            print(e)

        (rows,cols) = cv_image_d.shape

        img_n = cv2.normalize(src=cv_image_d, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        im_color = cv2.cvtColor(img_n, cv2.COLOR_GRAY2BGR)
        self.depth_img = im_color
        self.depth_raw = cv_image_d
    
    def pointcloud_callback(self, msg):
        self.pointcloud = msg

    def generate_img_detections(self, img, conf_thresh=0.85,show_imgs=False):
        '''Detect Objects with YOLOv5 Model'''

        detections = self.model.detect(img)
        output = []
        detections_img = img     

        if detections != []:
            for i, (x1, y1, x2, y2, cls_conf, i_label) in enumerate(detections):
                conf = cls_conf.item()

                if conf > conf_thresh:
                    _x = int(x1.item()) 
                    _y = int(y1.item())
                    _w = int(round(x2.item() - x1.item()))
                    _h = int(round(y2.item() - y1.item()))
                    label = str(self.names[int(i_label.item())]) 
                    output.append((_x,_y,_w,_h,label))

                    # Display de imagenes
                    label = label + '; conf=' + str(round(conf,3))
                    detections_img = cv2.rectangle(img, (_x, _y), (_x +_w, _y + _h), self.colors[int(i_label.item())], 2)
                    detections_img = cv2.putText(detections_img, label, (_x,_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[int(i_label.item())], 1, cv2.LINE_AA)
                    color = (255,0,0)

                    detections_img = cv2.circle(detections_img, (int(_x + _w/2), int(_y + _h/2)), radius=5, color=color, thickness=2)
                    print("detecte un " + str(label) + " con confianza de " + str(cls_conf.item()))

        # Mostrar Detecciones
        if show_imgs:
            cv2.imshow("Detecciones", detections_img)
            cv2.waitKey(10)

        # Publicacion de imagen con detecciones
        detections_img = self.bridge.cv2_to_imgmsg(detections_img, encoding="passthrough")
        self.image_pub.publish(detections_img)
        
        return output

    def generate_markers(self):
        while not rospy.is_shutdown():
            pcl = self.pointcloud
            img = self.img
            
            
            detections = self.generate_img_detections(img)

            if pcl is None or img is None or detections==[]: 
                # print('skipping')
                continue
            
            uvs_detections = []

            for (_x,_y,_w,_h,label) in detections:
                row, col = int(_x + _w/2), int(_y + _h/2)
                uvs_detections.append((row,col))
                
            # Procesamiento de Pointcloud
            points = pc2.read_points_list(pcl, skip_nans=True, uvs = uvs_detections)
            markers = MarkerArray()

            for i, point in enumerate(points):
                marker = Marker()
                marker.header.frame_id, marker.header.stamp = "camera_link", rospy.Time.now()

                # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
                marker.type, marker.id = 2, i

                # Set the scale and color of the marker
                marker.scale.x, marker.scale.y, marker.scale.z = 0.1, 0.1, 0.1
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 1.0, 0.0, 1.0

                # Set the pose of the marker and publish it 
                marker.pose.position.x = point.z    #  z ; z 
                marker.pose.position.y = - point.x  # -x ; y
                marker.pose.position.z = -point.y   #  y ; x 
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.lifetime.secs = 1
                markers.markers.append(marker)
            self.marker_pub.publish(markers)






if __name__ == '__main__':

    RS_MG = RealSense_Marker_Generator()
    print("RS_MG Node started!")
    
    try:
        RS_MG.generate_markers()

    except rospy.ROSInterruptException:
        rospy.logerr("ROS Interrupt Exception! Just ignore the exception!")