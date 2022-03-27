#pragma once

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
//  YOLO Bounding Boxes Header 
#include <darknet_ros_msgs/BoundingBoxes.h>
#include <eigen3/Eigen/Dense>
#include <dynamic_reconfigure/server.h>
// TODO: Add header for dynamic config file
#include <math.h>
#include <unordered_set>
#include "box_ekf.hpp"
// Dynamic reconfigure
#include <dynamic_reconfigure/server.h>
#include <yolo_ekf/YOLOEkfConfig.h>
// Message Filter for synchronizer 
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

namespace yolo_ekf{

// Exact Time Syncronizer for Image Messages and Yolo Bounding Boxes
typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, darknet_ros_msgs::BoundingBoxes> YoloSyncPolicy;


class yoloEkf {

public:
  yoloEkf(ros::NodeHandle n, ros::NodeHandle pn);

private:
  void recvSyncedBoxes(const sensor_msgs::ImageConstPtr& img_msg, const darknet_ros_msgs::BoundingBoxesConstPtr& bbox_msg);
  void reconfig(YOLOEkfConfig& config, uint32_t level);
  void timerCallback(const ros::TimerEvent& event);
  //void recvBboxes(const darknet_ros_msgs::BoundingBoxesConstPtr& bbox_msg);
  //void recvImgs(const sensor_msgs::ImageConstPtr& img_msg);
  void msgBox_to_ekfBox(const darknet_ros_msgs::BoundingBox& boundingbox, const ros::Time& boxStamp, filteredBox& ekfBox);
  double IoU(const filteredBox& detect_current, const filteredBox& detect_prev);
  int getUniqueId();

  // Setup for Image & bounding box synchronizer
  boost::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> sub_img_;
  boost::shared_ptr<message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes> > sub_objects_;
  boost::shared_ptr<message_filters::Synchronizer<YoloSyncPolicy>> sync_yolo_data_;
 
  ros::Subscriber sub_Bboxes_;
  ros::Publisher pub_ekf_boxes_;
  // TODO: Define some message that will publish the predicted bounding box states information as pub_box_comprehensive_
  ros::Publisher pub_box_comprehensive_;
  ros::Timer timer_;
  ros::Time previous_stamp;

  // for vizualization purposes
  ros::Subscriber sub_detectionimgs_;
  cv::Mat img_raw;
  std::vector<std::pair<int, cv::Rect2d>> cv_vects_;

  dynamic_reconfigure::Server<YOLOEkfConfig> srv_;
  YOLOEkfConfig cfg_;

  //Intersection Over Union Threshold
  double IoU_thresh;
  // Initialize vector to track ekf instances
  std::vector<boxEkf> box_ekfs_;

};

}