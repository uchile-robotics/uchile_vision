#!/usr/bin/env python
import time
import message_filters
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
class Inpainting():
    def __init__(self):

        # objects
        self.bridge = CvBridge()
        
        self.pub_inpainted = rospy.Publisher("/camera/depth_inpainted/image_rect_raw", Image, queue_size=3)
        self.pub_camera_info = rospy.Publisher("/camera/depth_inpainted/camera_info", CameraInfo, queue_size=3)

        #self.image_sub = message_filters.Subscriber("/depth_inpainted", Image)
        #self.info_sub = message_filters.Subscriber("/camera/depth/camera_info", CameraInfo)
        self.image_sub = rospy.Subscriber("/depth_inpainted", Image, self.imagecb)
        self.info_sub = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.infocb)
        #self.ts = message_filters.TimeSynchronizer([self.image_sub, self.info_sub], 3)
        #self.ts.registerCallback(self.callback)

        self.imgmsg = None
        self.infomsg= None

        # data
        self.depthimage = None

    def imagecb(self, msg):
        self.imgmsg = msg
    def infocb(self, msg):
        self.infomsg = msg

    def merge(self):
        now = rospy.Time.now()
        imgmsg = self.imgmsg
        infomsg = self.infomsg
        imgmsg.header.stamp = now
        infomsg.header.stamp = now
        self.pub_inpainted.publish(imgmsg)
        self.pub_camera_info.publish(infomsg)

if __name__=="__main__":
    try:
        rospy.init_node('inpainting')
        inpaint = Inpainting()
        time.sleep(3)
        while not rospy.is_shutdown():
            inpaint.merge()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass