#pragma once

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
//  YOLO Bounding Boxes Header 
#include <darknet_ros_msgs/BoundingBoxes.h>
#include <eigen3/Eigen/Dense>
#include <dynamic_reconfigure/server.h>
// TODO: Add header for dynamic config file
#include <math.h>
#include "box_ekf.hpp"

namespace yolo_ekf{


typedef Eigen::Matrix<double, 8, 1> StateVector;
typedef Eigen::Matrix<double, 8, 8> StateMatrix;


class yoloEkf {

public:
  yoloEkf(ros::NodeHandle n, ros::NodeHandle pn);

private:
  //TODO: Add dynamic configuration function for tuning q and r
  void timerCallback(const ros::TimerEvent& event);
  void recvBboxes(const darknet_ros_msgs::BoundingBoxesConstPtr& bbox_msg);
  void recvImgs(const sensor_msgs::ImageConstPtr& img_msg);
  void msgBox_to_ekfBox(const darknet_ros_msgs::BoundingBox& boundingbox, const ros::Time& boxStamp, filteredBox& ekfBox);
  double IoU(const filteredBox& detect_current, const filteredBox& detect_prev);
  int getUniqueId();
 
  ros::Subscriber sub_Bboxes_;
  ros::Publisher pub_ekf_boxes_;
  // TODO: Define some message that will publish the predicted bounding box states information as pub_box_comprehensive_
  ros::Publisher pub_box_comprehensive_;
  ros::Timer timer_;

  // for vizualization purposes
  ros::Subscriber sub_detectionimgs_;
  cv::Mat img_raw;
  std::vector<std::pair<int, cv::Rect2d>> cv_vects_;

  //dynamic_reconfigure::Server<paramer_configure_config_here> srv_;
  //<paramer_configure_config_here> cfg_;

  //Intersection Over Union Threshold
  double IoU_thresh;
  // Initialize vector to track ekf instances
  std::vector<boxEkf> box_ekfs_;

};

}