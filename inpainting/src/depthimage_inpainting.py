#!/usr/bin/env python
import time
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
class Inpainting():
    def __init__(self):

        # objects
        self.bridge = CvBridge()
        sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        pub = rospy.Publisher("/camera/depth/inpainted", Image)

        # data
        self.depthimage = None

    def depth_callback(self, img_raw):
        self.depthimage = self.bridge.imgmsg_to_cv2(img_raw, desired_encoding='passthrough')

    def showImg(self):
        cv2.imshow('image', self.depthimage)
        cv2.waitKey(1)

    def inpaint(self):
        # TODO: Aplicar canny para generar una mascara de los bordes, 8bit input image. Que tenemos nosotros?
        depth_image = self.depthimage
        edges = cv2.Canny(depth_image, 100, 200)

        # TODO: Con la mascara y la imagen, aplicar inpainting
        dst = cv2.inpaint(depth_image, edges, 3, cv2.INPAINT_TELEA)

        # TODO: Transformar la imagen inpainted a sensor_msgs.msg.Image

        # TODO: Publicar la imagen correcta
        pass


if __name__=="__main__":
    try:
        rospy.init_node('inpainting')
        inpaint = Inpainting()
        time.sleep(1)
        while not rospy.is_shutdown():
            inpaint.inpaint()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass