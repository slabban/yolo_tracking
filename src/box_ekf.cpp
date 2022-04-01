#include "box_ekf.hpp"

namespace yolo_ekf{

  boxEkf::boxEkf(filteredBox detection)
  {
    // Initialize state estimate to input arguments
    X_ << detection.cx, detection.cy, detection.width, detection.height, detection.vx, detection.vy, detection.vw, detection.vh;

    estimate_stamp_ = detection.stamp;
    measurement_stamp_ = detection.stamp;
    spawn_stamp_ = detection.stamp;
    filteredBox_ = detection;

  }

  // Implement State Prediction Step
  StateVector boxEkf::statePrediction(double dt, const StateVector& old_state){

    double cx = old_state(0);
    double cy = old_state(1);
    double width = old_state(2);
    double height = old_state(3);
    double vx = old_state(4);
    double vy = old_state(5);
    double vw = old_state(6);
    double vh = old_state(7);

    StateVector new_state;
    new_state(0) = cx + (dt*vx);
    new_state(1) = cy + (dt*vy);
    new_state(2) = width + (dt*vw);
    new_state(3) = height + (dt*vh);
    new_state(4) = vx;
    new_state(5) = vy;
    new_state(6) = vw;
    new_state(7) = vh;
    return new_state;
  }

  // Populate state Jacobian with current state values
  StateMatrix boxEkf::stateJacobian(double dt, const StateVector& state){

    StateMatrix A;
    A.row(0) << 1,0,0,0,dt,0,0,0;
    A.row(1) << 0,1,0,0,0,dt,0,0;
    A.row(2) << 0,0,1,0,0,0,dt,0;
    A.row(3) << 0,0,0,1,0,0,0,dt;
    A.row(4) << 0,0,0,0,1,0,0,0;
    A.row(5) << 0,0,0,0,0,1,0,0;
    A.row(6) << 0,0,0,0,0,0,1,0;
    A.row(7) << 0,0,0,0,0,0,0,1;
    return A;
  }

  //Propagate covariance matrix by one step
  StateMatrix boxEkf::covPrediction(const StateMatrix& A, const StateMatrix& Q, const StateMatrix& old_cov){
    StateMatrix new_cov;
    new_cov = A*old_cov*A.transpose() + Q;
    return new_cov;
  }


  void boxEkf::updateFilterPredict(const ros::Time& current_time){

  if(estimate_stamp_ == ros::Time(0)){
    ROS_WARN_THROTTLE(1.0, "Waiting for the first Detections, ignoring this update...");
    return;
  }

  // Compute amount of time to advance the state prediction
  double dt = (current_time - estimate_stamp_).toSec();

  // Propagate estimat prediction and update estimate with result
  StateMatrix A = stateJacobian(dt, X_);
  X_ = statePrediction(dt, X_);
  P_ = covPrediction(A, Q_, P_);
  estimate_stamp_ = current_time;

  // TODO: insert boundaries based on total image size to prevent bounding box estimates that are out 
  // of the image pixel frame
}

void boxEkf::updateFilterMeasurement(const ros::Time& current_time, const filteredBox& ekf_bounding_box){

  // Calculate time difference between measurement and filter state
  double dt = (current_time - estimate_stamp_).toSec();
  //ROS_INFO("Detections update delta t: %f seconds", dt);

  if (fabs(dt) > 2) {
    // Large time jump detected... reset filter to this measurement
    ROS_INFO("Large time jump detected... resetting filter to this measurement");
    X_ << ekf_bounding_box.cx, ekf_bounding_box.cy, ekf_bounding_box.width, ekf_bounding_box.height, 0, 0, 0, 0;
    P_.setIdentity();
    //spawn_stamp_ = meas.header.stamp;
    estimate_stamp_ = current_time;
    measurement_stamp_ = current_time;
    return;
  }

  filteredBox_.darknet_box.xmin = ekf_bounding_box.darknet_box.xmin;
  filteredBox_.darknet_box.xmax = ekf_bounding_box.darknet_box.xmax;
  filteredBox_.darknet_box.ymin = ekf_bounding_box.darknet_box.ymin;
  filteredBox_.darknet_box.ymax = ekf_bounding_box.darknet_box.ymax;

  // Prediction step
  StateMatrix A = stateJacobian(dt, X_);
  StateVector predicted_state = statePrediction(dt, X_);
  StateMatrix predicted_cov = covPrediction(A, Q_, P_); 

  // Measurement update
  // Define measurement matrix 
  Eigen::Matrix<double, 4, 8> C;
  C.row(0) << 1,0,0,0,0,0,0,0;
  C.row(1) << 0,1,0,0,0,0,0,0;
  C.row(2) << 0,0,1,0,0,0,0,0;
  C.row(3) << 0,0,0,1,0,0,0,0;

  // Compute expected measurement based on predicted_state 
  Eigen::Vector4d expected_meas;
  expected_meas << predicted_state(0), predicted_state(1), predicted_state(2), predicted_state(3);

  // Put bounding box measurment in eigen object
  Eigen::Vector4d real_meas;
  real_meas << ekf_bounding_box.cx, ekf_bounding_box.cy, ekf_bounding_box.width, ekf_bounding_box.height;

  // // Define Measurement Noise R
  // Eigen::Matrix4d R_;
  // R_.row(0) << 1,0,0,0;
  // R_.row(1) << 0,1,0,0;
  // R_.row(2) << 0,0,10,0;
  // R_.row(3) << 0,0,0,10;

  // Compute residual(Innovation) covariance matrix 
  Eigen::Matrix4d S;
  S = C * predicted_cov * C.transpose() + R_;

  // Compute Kalman Gain Matrix
  // The K matrix dimensions are deduced by computing the matrix in the Kalman gain formula below
  Eigen::Matrix<double, 8, 4> K;
  K = predicted_cov*C.transpose()*S.inverse();

  // Update state estimate based on difference between actual and expected measurement 
  X_ = predicted_state+ K*(real_meas - expected_meas);

  // Update estimate error covariance using Kalman gain matrix
  P_ = (StateMatrix::Identity() - K*C)*predicted_cov;

  // Set estimate time stamp and latest measurement time stamp to the stamp in the input argument
  estimate_stamp_ = current_time;
  measurement_stamp_ = current_time;

}

filteredBox boxEkf::getfilteredBox(){
  return filteredBox_;
}

filteredBox boxEkf::getEstimate(){
  filteredBox estimate_output;
  estimate_output.cx = X_(0);
  estimate_output.cy = X_(1);
  estimate_output.width = X_(2);
  estimate_output.height = X_(3);
  estimate_output.vx = X_(4);
  estimate_output.vy = X_(5);
  estimate_output.vw = X_(6);
  estimate_output.vh = X_(7);
  estimate_output.id = filteredBox_.id;

  estimate_output.darknet_box.xmin = estimate_output.cx - (0.5*estimate_output.width);
  estimate_output.darknet_box.xmax = estimate_output.cx + (0.5*estimate_output.width);
  estimate_output.darknet_box.ymin = estimate_output.cy - (0.5*estimate_output.height);
  estimate_output.darknet_box.ymax = estimate_output.cy + (0.5*estimate_output.height);
  estimate_output.darknet_box.Class = filteredBox_.darknet_box.Class;

  return estimate_output;
}

int boxEkf::getId()
{
  return id_ = filteredBox_.id;
}

bool boxEkf::isStale(float max_age) {
  return (estimate_stamp_ - measurement_stamp_) > ros::Duration(max_age);
}

double boxEkf::getAge(){
  return (estimate_stamp_ - spawn_stamp_).toSec();
}

// Sets the process noise standard deviations
void boxEkf::setQ(double q_pos, double q_vel)
{
  // Populate Q_ with q_pos and q_vel
  Q_.setZero();
  Q_(0, 0) = q_pos;
  Q_(1, 1) = q_pos;
  Q_(2, 2) = q_pos;
  Q_(3, 3) = q_pos;
  Q_(4, 4) = q_vel;
  Q_(5, 5) = q_vel;
  Q_(6, 6) = q_vel;
  Q_(7, 7) = q_vel;

  // Q_.row(0) << q_pos,q_pos,0,0,0,0,0,0;
  // Q_.row(1) << q_pos,q_pos,0,0,0,0,0,0;
  // Q_.row(2) << 0,0,q_pos,q_pos,0,0,0,0;
  // Q_.row(3) << 0,0,q_pos,q_pos,0,0,0,0;
  // Q_.row(4) << 0,0,0,0,q_pos,q_pos,0,0;
  // Q_.row(5) << 0,0,0,0,q_pos,q_vel,0,0;
  // Q_.row(6) << 0,0,0,0,0,0,q_vel,q_pos;
  // Q_.row(7) << 0,0,0,0,0,0,q_pos,q_vel;
}

// Sets the measurement noise standard deviation
void boxEkf::setR(double r_cx_cy, double r_width_height)
{
  // Populate R_ with r_pos
  R_.setZero();
  R_(0, 0) = r_cx_cy;
  R_(1, 1) = r_cx_cy;
  R_(2, 2) = r_width_height;
  R_(3, 3) = r_width_height;
}

void boxEkf::setP(double p_init){
  // Initialize the covariance matrix
  P_.setIdentity() * p_init;
}


}