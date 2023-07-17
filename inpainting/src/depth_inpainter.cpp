#include "test_inpaint.cpp"
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

static const std::string OPENCV_WINDOW = "Image window";
const unsigned char noDepth = 0;

class DepthInpainter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;

  image_transport::CameraSubscriber cam_sub_;
  image_transport::CameraPublisher cam_pub_;

public:
  DepthInpainter()
    : it_(nh_)
  {
    // Subscribe to input video feed and publish output video feed
    //image_sub_ = it_.subscribe("/camera/depth/image_rect_raw", 1,
    //  &DepthInpainter::imageCb, this);
    //image_pub_ = it_.advertise("/depth_inpainted", 1);

    cam_sub_ = it_.subscribeCamera("camera/depth/image_rect_raw", 3, &DepthInpainter::camera_Cb, this);
    cam_pub_ = it_.advertiseCamera("/camera/depth_inpainted/image_rect_raw",3);

    cv::namedWindow(OPENCV_WINDOW);
  }

  ~DepthInpainter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }


  void camera_Cb(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::CameraInfoConstPtr& cam_info)
  {
    
    cv_bridge::CvImagePtr cv_ptr;
    cv::Mat temp;

    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
  
    custom::inpaint2(cv_ptr->image, (cv_ptr->image == noDepth), cv_ptr->image, 3.0, cv::INPAINT_TELEA);

    // Output modified video stream
    // image_pub_.publish(cv_ptr->toImageMsg());
    cam_pub_.publish(cv_ptr->toImageMsg(), cam_info);
    //cam_pub_.publish(cv_ptr, cam_info);
  }

};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "depth_inpainter");
  DepthInpainter ic;
  ros::spin();
  return 0;
}