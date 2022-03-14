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

  // Prediction Step of the EKF
  void updateFilterPredict(const ros::Time& current_time);
  // Update Step of the EKF
  void updateFilterMeasurement(const ros::Time& current_time, const darknet_ros_msgs::BoundingBox& boundingbox);

  void msgBox_to_ekfBox(const darknet_ros_msgs::BoundingBox& boundingbox, const ros::Time& boxStamp, filteredBox& ekfBox);
  void ekfBox_to_msgBox(const filteredBox& ekfBox, darknet_ros_msgs::BoundingBox& boundingbox);

  double IoU(const filteredBox& detect_current, const filteredBox& detect_prev, const double& IoU_thresh);
  int getUniqueId();

  // Predict State
  StateVector statePrediction(double dt, const StateVector& old_state);
  // Update State Jacobian
  StateMatrix stateJacobian(double dt, const StateVector& state);
  //Predict Next Covariance
  StateMatrix covPrediction(const StateMatrix& A, const StateMatrix& Q, const StateMatrix& old_cov);

  ros::Subscriber sub_Bboxes_;
  ros::Publisher pub_ekf_boxes_;
  // TODO: Define some message that will publish the predicted bounding box states information as pub_box_comprehensive_
  ros::Publisher pub_box_comprehensive_;
  ros::Timer timer_;

  //dynamic_reconfigure::Server<paramer_configure_config_here> srv_;
  //<paramer_configure_config_here> cfg_;

  // Estimate State, Covariance, and Current Timestamp
  StateVector X_;
  StateMatrix P_;
  ros::Time estimate_stamp_;
  // Process Noise Covariance
  StateMatrix Q_;

  //Intersection Over Union Threshold
  double IoU_thresh;
  // Initialize vector to track ekf instances
  std::deque<boxEkf> box_ekfs_;

};

}