#pragma once

#include <ros/ros.h>
//  YOLO Bounding Boxes Header 
#include <darknet_ros_msgs/BoundingBoxes.h>
#include <eigen3/Eigen/Dense>
#include <dynamic_reconfigure/server.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>


namespace yolo_ekf{

// Define a struct that contains the states to be used in the EKF
typedef struct 
{ darknet_ros_msgs::BoundingBox darknet_box;
  double cx;
  double cy;
  double width;
  double height;
  double vx;
  double vy;
  double vw;
  double vh;
  ros::Time stamp;
}filteredBox;

typedef Eigen::Matrix<double, 8, 1> StateVector;
typedef Eigen::Matrix<double, 8, 8> StateMatrix;

class boxEkf{

  public:

  boxEkf(filteredBox detection);

  // Prediction Step of the EKF
  void updateFilterPredict(const ros::Time& current_time);
  // Update Step of the EKF
  void updateFilterMeasurement(const ros::Time& current_time, const filteredBox& ekf_bounding_box);
  // Getter function for EKF instance filtered box
  filteredBox getfilteredBox();
  // getter function for EKF instance Id
  int getId();
  bool isStale();

  private:

  // Predict State
  StateVector statePrediction(double dt, const StateVector& old_state);
  // Update State Jacobian
  StateMatrix stateJacobian(double dt, const StateVector& state);
  //Predict Next Covariance
  StateMatrix covPrediction(const StateMatrix& A, const StateMatrix& Q, const StateMatrix& old_cov);

  // Estimate State, Covariance, and Current Timestamp
  StateVector X_;
  StateMatrix P_;
  ros::Time estimate_stamp_;

  // Timestamp from last measurement
  ros::Time measurement_stamp_;

  // Process Noise Covariance
  StateMatrix Q_;

  //ekf filtered box
  filteredBox filteredBox_;

  //ekf Unique Id
  // Return the ID number property
  int id_;
};

}