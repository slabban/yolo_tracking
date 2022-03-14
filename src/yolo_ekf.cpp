#include "yolo_ekf.hpp"

using namespace Eigen;


namespace yolo_ekf{

yoloEkf::yoloEkf(ros::NodeHandle n, ros::NodeHandle pn){

  double sample_time = 0.1;

  sub_Bboxes_ = n.subscribe("/darknet_ros/bounding_boxes", 1 , &yoloEkf::recvBboxes, this);
  pub_ekf_boxes_ = n.advertise<darknet_ros_msgs::BoundingBoxes>("boxes_ekf", 1);
  timer_ = n.createTimer(ros::Duration(sample_time), &yoloEkf::timerCallback, this);
  //TODO: Setup Dynamic Config Server here

  X_.setZero();
  P_ = P_.setIdentity() * 10;

  IoU_thresh = 0.33;

}
void yoloEkf::timerCallback(const ros::TimerEvent& event){

  // TODO: Implement Stale object checker by 

  updateFilterPredict(event.current_real);

  darknet_ros_msgs::BoundingBoxes bbox_tracks;

  filteredBox ekfBox_tracks;

  ekfBox_tracks.cx = X_(0);
  ekfBox_tracks.cy = X_(1);
  ekfBox_tracks.width = X_(2);
  ekfBox_tracks.height = X_(3);

}

void yoloEkf::recvBboxes(const darknet_ros_msgs::BoundingBoxesConstPtr& bbox_msg){

  // Loop through the incoming bounding boxes
  // TODO: This process either initializes a new kalman filter instance or run the update measurment function
  //       based on an association algorithm that will attempt to match the boxes(t) to boxes(t-1)
  //       Unmatched boxes will be instantiated, and matched boxes willbe updated. 

  // Vector to hold the EKF indices that have already been matched to an incoming object measurement
  std::vector<int> matched_detection_indices;
  // Vector to hold array indices of objects to create new EKF instances from
  std::vector<filteredBox> new_detection_indices;

  for (auto& bounding_box : bbox_msg->bounding_boxes){
    for(size_t i=0; i<box_ekfs_.size(); ++i){

      filteredBox current_box;
      msgBox_to_ekfBox(bounding_box, bbox_msg->header.stamp, current_box);
      
      int IoU_score = IoU(current_box, box_ekfs_[i].getfilteredBox(), IoU_thresh);

      // If IoU passes and the classes match, we can associate the incoming detection with the existing ekf instance
      if (IoU_score >= IoU_thresh && current_box.darknet_box.Class == box_ekfs_[i].getfilteredBox().darknet_box.Class){
        box_ekfs_[i].updateFilterMeasurement(current_box.stamp, current_box);
      }
      else{
        new_detection_indices.push_back(current_box);
      }
    }
    // After trying to associate all incoming object measurements to existing EKF instances,
    // create new EKF instances to track the inputs that weren't associated with existing ones
    for (auto new_object : new_detection_indices) {

      box_ekfs_.push_back(boxEkf(new_object));
      // object_ekfs_.back().setQ(cfg_.q_pos, cfg_.q_vel);
      // object_ekfs_.back().setR(cfg_.r_pos);
    }
  }
}



// Implement State Prediction Step
StateVector yoloEkf::statePrediction(double dt, const StateVector& old_state){

  int cx = old_state(0);
  int cy = old_state(1);
  int width = old_state(2);
  int height = old_state(3);
  float vx = old_state(4);
  float vy = old_state(5);
  float vw = old_state(6);
  float vh = old_state(7);

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
StateMatrix yoloEkf::stateJacobian(double dt, const StateVector& state){

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
StateMatrix yoloEkf::covPrediction(const StateMatrix& A, const StateMatrix& Q, const StateMatrix& old_cov){

  StateMatrix new_cov;
  new_cov = A*old_cov*A.transpose() + Q;
  return new_cov;
}

void yoloEkf::updateFilterPredict(const ros::Time& current_time){

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

void yoloEkf::updateFilterMeasurement(const ros::Time& current_time, const darknet_ros_msgs::BoundingBox& boundingbox){

  filteredBox ekf_bounding_box = {};
  //msgBox_to_ekfBox(boundingbox, ekf_bounding_box);  

  // Calculate time difference between measurement and filter state
  double dt = (current_time - estimate_stamp_).toSec();
  ROS_INFO("Detections update delta t: %f seconds", dt);

  if (fabs(dt) > 2) {
    // Large time jump detected... reset filter to this measurement
    X_ << ekf_bounding_box.cx, ekf_bounding_box.cy, ekf_bounding_box.width, ekf_bounding_box.height, 0, 0, 0, 0;
    P_.setIdentity();
    //spawn_stamp_ = meas.header.stamp;
    estimate_stamp_ = current_time;
    //measurement_stamp_ = current_time;
    return;
  }

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

  // Define Measurement Noise R
  Eigen::Matrix4d R_;
  R_.row(0) << 1,0,0,0;
  R_.row(1) << 0,1,0,0;
  R_.row(2) << 0,0,10,0;
  R_.row(3) << 0,0,0,10;

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

  estimate_stamp_ = current_time;

}

void yoloEkf::msgBox_to_ekfBox(const darknet_ros_msgs::BoundingBox& boundingbox, const ros::Time& boxStamp, filteredBox& ekfBox){

  ekfBox.cx = 0.5*(boundingbox.xmax + boundingbox.xmin);
  ekfBox.cy = 0.5*(boundingbox.ymax + boundingbox.ymin);
  ekfBox.width = boundingbox.xmax - boundingbox.xmin;
  ekfBox.height = boundingbox.ymax - boundingbox.ymin;
  ekfBox.vx = 0;
  ekfBox.vy = 0;
  ekfBox.vw = 0;
  ekfBox.vh = 0;
  ekfBox.stamp = boxStamp;
  ekfBox.darknet_box = boundingbox;
}

void yoloEkf::ekfBox_to_msgBox(const filteredBox& ekfBox, darknet_ros_msgs::BoundingBox& boundingbox){

}


double yoloEkf::IoU(const filteredBox& detect_current, const filteredBox& detect_prev, const double& IoU_thresh)
{  
  double r1_xmin = detect_current.darknet_box.xmin;
  double r1_ymin = detect_current.darknet_box.ymin;
  double r1_xmax = detect_current.darknet_box.xmax;
  double r1_ymax = detect_current.darknet_box.ymax;
  double r2_xmin = detect_prev.darknet_box.xmin;
  double r2_ymin = detect_prev.darknet_box.ymin;
  double r2_xmax = detect_prev.darknet_box.xmax;
  double r2_ymax = detect_prev.darknet_box.ymax;
  //double width = (detect.xmax-detect.xmin);
  //double height =(detect.ymax-detect.ymin);

  // If one rectangle is on left side of other 
  if (r1_xmin >= r2_xmax || r2_xmin >= r1_xmax)
  {
   return -1;
  }     
   //If one rectangle is above other (Recall down is positive in OpenCV)
  if (r1_ymin >= r2_ymax || r2_ymin >= r1_ymax) 
  {
    return -1;
  }

  //Area of rectangles
  double r1_area = detect_current.width * detect_current.height;
  //double r2_area = r2.area();
  double r2_area = detect_prev.width * detect_prev.height;
  double ri_x = cv::max(r1_xmin, r2_xmin);
  double ri_y = cv::max(r1_ymin, r2_ymin);
  //Locate top-right of intersected rectangle
  double ri_xmax = cv::min(r1_xmax,r2_xmax);
  double ri_ymax = cv::min(r1_ymax,r2_ymax);
  //Calculate Intersected Width
  double ri_width = ri_xmax - ri_x;
  //Calculate Intersected Height
  double ri_height = ri_ymax - ri_y;
  //Area of intersection
  double ri_area = ri_height*ri_width;
  //Determine Intersection over Union
  double IoU = ri_area/((r1_area + r2_area)-ri_area);

  return IoU;
}

int yoloEkf::getUniqueId()
  {
//return id ++ 
    int id = 0;
    bool done = false;
    while (!done) {
      done = true;
      for (auto& track : box_ekfs_) {
        if (track.getId() == id) {
          done = false;
          id++;
          break;
        }
      }
    }
    return id;
  }


}



