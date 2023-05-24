#!/usr/bin/env python
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
def Inpainting():
    def __innit__(self):
        # subscribers and publishers
        sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)

        # datos
        self.depthimage = None

    def depth_callback(self, img_raw):
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(img_raw, desired_encoding='passthrough')
        self.depthimage = cv_image
    def showImg(self):
        cv2.imshow('image',self.depthimage)
        cv2.waitKey(0)
if __name__=="__main__":
    try:
        inpaint = Inpainting()
    except rospy.ROSInterruptException:
        pass