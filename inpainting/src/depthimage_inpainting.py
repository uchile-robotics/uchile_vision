#!/usr/bin/env python
import time
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
class Inpainting():
    def __init__(self):

        # objects
        self.bridge = CvBridge()
        self.sub_depth = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)
        self.sub_info  = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.camerainfo_callback)
        self.pub = rospy.Publisher("/camera/depth_inpainted/image_rect_raw", Image, queue_size=1)
        self.pub_info = rospy.Publisher("/camera/depth_inpainted/camera_info", CameraInfo, queue_size=1)

        # data
        self.depthimage = None
        self.camerainfo = None

    def depth_callback(self, img_raw):
        self.depthimage = self.bridge.imgmsg_to_cv2(img_raw, desired_encoding='passthrough')

    def camerainfo_callback(self, camerainfo):
        self.camerainfo = camerainfo

    def showImg(self):
        cv2.imshow('image', self.depthimage)
        cv2.waitKey(1)

    def inpaint(self):
        # TODO: Aplicar canny para generar una mascara de los bordes, 8bit input image. Que tenemos nosotros?
        depth_image = self.depthimage
        ratio = np.amax(depth_image) / 256
        depth_image = (depth_image/ratio).astype('uint8')
        _, edges = cv2.threshold(depth_image, 0, 255, cv2.THRESH_BINARY_INV)
        dst = cv2.inpaint(depth_image, edges, 3, cv2.INPAINT_TELEA)
        dst = (dst).astype('uint16')*ratio
        msg = self.bridge.cv2_to_imgmsg(dst, encoding="passthrough")
        self.pub.publish(msg)
        self.pub_info.publish(self.camerainfo)

if __name__=="__main__":
    try:
        rospy.init_node('inpainting')
        time.sleep(1)
        inpaint = Inpainting()
        time.sleep(1)
        while not rospy.is_shutdown():
            inpaint.inpaint()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass