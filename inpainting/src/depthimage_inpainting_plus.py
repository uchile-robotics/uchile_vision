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
        
        self.pub_inpainted = rospy.Publisher("/camera/depth_inpainted/camera_rect_raw", Image, queue_size=3)
        self.pub_camera_info = rospy.Publisher("/camera/depth_inpainted/camera_info", CameraInfo, queue_size=3)

        self.image_sub = message_filters.Subscriber("/camera/depth/image_rect_raw", Image)
        self.info_sub = message_filters.Subscriber("/camera/depth/camera_info", CameraInfo)

        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.info_sub], 3)
        self.ts.registerCallback(self.inpaint_callback)

        # data
        self.depthimage = None

    def inpaint_callback(self, image, camera_info):
        image = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        self.inpaint(depth_image=image, camera_info=camera_info)

    def depth_callback(self, img_raw):
        self.depthimage = self.bridge.imgmsg_to_cv2(img_raw, desired_encoding='passthrough')

    def showImg(self):
        cv2.imshow('image', self.depthimage)
        cv2.waitKey(1)

    def inpaint(self, depth_image, camera_info):
        ratio = np.amax(depth_image) / 256
        depth_image = (depth_image/ratio).astype('uint8')
        _, edges = cv2.threshold(depth_image, 0, 255, cv2.THRESH_BINARY_INV)
        dst = cv2.inpaint(depth_image, edges, 3, cv2.INPAINT_TELEA)
        dst = (dst).astype('uint16')*ratio
        msg = self.bridge.cv2_to_imgmsg(dst, encoding="passthrough")

        now = rospy.Time.now()
        msg.header.stamp = now
        camera_info.header.stamp = now
        self.pub_inpainted.publish(msg)
        self.pub_camera_info.publish(camera_info)

if __name__=="__main__":
    try:
        rospy.init_node('inpainting')
        inpaint = Inpainting()
        time.sleep(1)
        #while not rospy.is_shutdown():
        #    inpaint.inpaint()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass