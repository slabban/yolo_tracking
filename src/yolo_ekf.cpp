#include "yolo_ekf.hpp"


using namespace Eigen;


namespace yolo_ekf{

yoloEkf::yoloEkf(ros::NodeHandle n, ros::NodeHandle pn){

  double sample_time = 0.02;

  sub_detectionimgs_ = n.subscribe("/darknet_ros/detection_image", 1, &yoloEkf::recvImgs, this);
  sub_Bboxes_ = n.subscribe("/darknet_ros/bounding_boxes", 1 , &yoloEkf::recvBboxes, this);
  pub_ekf_boxes_ = n.advertise<darknet_ros_msgs::BoundingBoxes>("boxes_ekf", 1);
  timer_ = n.createTimer(ros::Duration(sample_time), &yoloEkf::timerCallback, this);
  //TODO: Setup Dynamic Config Server here

  // IoU threshold for box association algorithms
  IoU_thresh = 0.5;
  cv::namedWindow("Sync_Output", cv::WINDOW_NORMAL);
}

void yoloEkf::recvImgs(const sensor_msgs::ImageConstPtr& img_msg){
    img_raw = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8)->image;
    for(size_t i = 0; i < cv_vects_.size(); ++i){
      cv::Point2d corner((cv_vects_[i].second.x + cv_vects_[i].second.width), cv_vects_[i].second.y);
      cv::rectangle(img_raw, cv_vects_[i].second, cv::Scalar(0,255,0));
      cv::putText(img_raw, std::to_string(cv_vects_[i].first), corner, cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0));
    }
    imshow("Sync_Output", img_raw);
    cv::waitKey(1);
}


void yoloEkf::timerCallback(const ros::TimerEvent& event){
  // Delete stale objects that have not been observed for a while
  std::vector<size_t> stale_objects;
  for (size_t i = 0; i < box_ekfs_.size(); ++i) {
    box_ekfs_[i].updateFilterPredict(event.current_real);
    if (box_ekfs_[i].isStale()) {
      stale_objects.push_back(i);
    }
  }
  for (int i = (int)stale_objects.size() - 1; i >= 0; i--) {
    box_ekfs_.erase(box_ekfs_.begin() + stale_objects[i]);
  }
  // Loop through the estimates and estimated bounding boxes on to cv image
  cv_vects_.clear();
  for (size_t i = 0; i < box_ekfs_.size(); ++i) {
    filteredBox estimated_output = box_ekfs_[i].getEstimate();
    cv::Rect2d prediction_cv(estimated_output.darknet_box.xmin, estimated_output.darknet_box.ymin,
    estimated_output.width,estimated_output.height);
    cv_vects_.push_back({estimated_output.id,prediction_cv});
  }
}

// This callback either initializes a new kalman filter instance or runs the update measurment function
// based on an association algorithm that will attempt to match the boxes(t) to boxes(t-1)
// Unmatched boxes will be instantiated, and matched boxes willbe updated. 
void yoloEkf::recvBboxes(const darknet_ros_msgs::BoundingBoxesConstPtr& bbox_msg){


  // unordered set to hold the EKF indices that have already been matched to an incoming object measurement
  std::unordered_set<int> matched_detection_indices;
  // Vector to hold array indices of objects to create new EKF instances from
  std::vector<filteredBox> new_detection_boxes;

  // Find closest candidate with the highest IoU
  for (auto& bounding_box : bbox_msg->bounding_boxes){

    // Exclude truck detections due to class confusion limited Yolo performance
    if(bounding_box.Class == "truck"){
      continue;
    }
    filteredBox current_box;
    msgBox_to_ekfBox(bounding_box, bbox_msg->header.stamp, current_box);

    filteredBox* current_candidate = nullptr;
    boxEkf* associated_filter = nullptr;
    double IoU_score_current = -1.f;
    double IoU_score_max = -1.f;

    for(size_t i=0; i<box_ekfs_.size(); ++i){

      // Check to see if the current index has already been matched, if so, skip this iteration
      auto hasMatch = matched_detection_indices.find(box_ekfs_[i].getId());
      if(hasMatch != matched_detection_indices.end()){
        continue;
      }

      IoU_score_current = IoU(current_box, box_ekfs_[i].getEstimate());
      //ROS_INFO("current IoU is: %f", IoU_score_current);
      //&& current_box.darknet_box.Class == box_ekfs_[i].getEstimate().darknet_box.Class
      if(IoU_score_current > IoU_score_max){
        current_candidate = &current_box;
        current_candidate->id = box_ekfs_[i].getId();
        associated_filter = &box_ekfs_[i];
        IoU_score_max = IoU_score_current;
      }
    }
    //ROS_INFO("Max IoU is: %f", IoU_score_max);
    // If highest IoU passes and the classes match, we can associate the incoming detection with the existing ekf instance
    // and add the index to the matched set
    if (IoU_score_max >= IoU_thresh){
      associated_filter->updateFilterMeasurement(current_candidate->stamp, *current_candidate);
      matched_detection_indices.insert(current_candidate->id);
    }
    else{
      new_detection_boxes.push_back(current_box);
    }
  }
  // create new EKF instances to track the inputs that weren't associated with existing ones
  for (auto new_object : new_detection_boxes) {
    new_object.id = getUniqueId();
    box_ekfs_.push_back(boxEkf(new_object));
    }
}

// This converts the incoming darknet bounding boxes to the 'filteredbox' struct 
void yoloEkf::msgBox_to_ekfBox(const darknet_ros_msgs::BoundingBox& boundingbox, const ros::Time& boxStamp, filteredBox& ekfBox){
  ekfBox.cx = 0.5*(boundingbox.xmax + boundingbox.xmin);
  ekfBox.cy = 0.5*(boundingbox.ymax + boundingbox.ymin);
  ekfBox.width = boundingbox.xmax - boundingbox.xmin;
  ekfBox.height = boundingbox.ymax - boundingbox.ymin;
  ekfBox.vx = 0;
  ekfBox.vy = 0;
  ekfBox.vw = 0;
  ekfBox.vh = 0;
  ekfBox.id = -1;
  ekfBox.stamp = boxStamp;
  ekfBox.darknet_box = boundingbox;
}

double yoloEkf::IoU(const filteredBox& detect_current, const filteredBox& detect_prev)
{  
  // wrapping box names to r1 and r2 for readability
  double r1_xmin = detect_current.darknet_box.xmin;
  double r1_ymin = detect_current.darknet_box.ymin;
  double r1_xmax = detect_current.darknet_box.xmax;
  double r1_ymax = detect_current.darknet_box.ymax;
  double r2_xmin = detect_prev.darknet_box.xmin;
  double r2_ymin = detect_prev.darknet_box.ymin;
  double r2_xmax = detect_prev.darknet_box.xmax;
  double r2_ymax = detect_prev.darknet_box.ymax;
  double r1_width = r1_xmax - r1_xmin;
  double r1_height = r1_ymax - r1_ymin;
  double r2_width = r2_xmax - r2_xmin;
  double r2_height = r2_ymax - r2_ymin;
  //If one rectangle is on left side of other 
  if (r1_xmin >= r2_xmax || r2_xmin >= r1_xmax)
  {
    //ROS_INFO("Trigger 1");
   return -1.0;
  }     
   //If one rectangle is above other (Recall down is positive in OpenCV)
  if (r1_ymin >= r2_ymax || r2_ymin >= r1_ymax) 
  {
    //ROS_INFO("Trigger 2");
    return -1.0;
  }
  //Area of rectangles
  double r1_area = r1_width * r1_height;
  double r2_area = r2_width * r2_height;
  double ri_x = std::max(r1_xmin, r2_xmin);
  double ri_y = std::max(r1_ymin, r2_ymin);
  //Locate top-right of intersected rectangle
  double ri_xmax = std::min(r1_xmax,r2_xmax);
  double ri_ymax = std::min(r1_ymax,r2_ymax);
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



