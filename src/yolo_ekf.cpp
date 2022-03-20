#include "yolo_ekf.hpp"


using namespace Eigen;


namespace yolo_ekf{

yoloEkf::yoloEkf(ros::NodeHandle n, ros::NodeHandle pn){

  double sample_time = 0.1;

  sub_detectionimgs_ = n.subscribe("/darknet_ros/detection_image", 1, &yoloEkf::recvImgs, this);
  sub_Bboxes_ = n.subscribe("/darknet_ros/bounding_boxes", 1 , &yoloEkf::recvBboxes, this);
  pub_ekf_boxes_ = n.advertise<darknet_ros_msgs::BoundingBoxes>("boxes_ekf", 1);
  timer_ = n.createTimer(ros::Duration(sample_time), &yoloEkf::timerCallback, this);
  //TODO: Setup Dynamic Config Server here

  IoU_thresh = 0.33;

  cv::namedWindow("Sync_Output", cv::WINDOW_NORMAL);
}

void yoloEkf::recvImgs(const sensor_msgs::ImageConstPtr& img_msg){
    img_raw = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8)->image;
    cv::pyrDown(img_raw, img_raw, cv::Size(img_raw.cols/2, img_raw.rows/2));
    cv::rectangle(img_raw, test, cv::Scalar(0,0,255));
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

  for (size_t i = 0; i < box_ekfs_.size(); ++i) {

    //ROS_INFO("loop!");

    filteredBox estimated_output = box_ekfs_[i].getEstimate();


    cv::Rect2d prediction_cv(estimated_output.cx - (0.5*estimated_output.width),
    estimated_output.cy - (0.5*estimated_output.height),
    estimated_output.width,estimated_output.height);

    test = prediction_cv;

  }


}

void yoloEkf::recvBboxes(const darknet_ros_msgs::BoundingBoxesConstPtr& bbox_msg){

  // Loop through the incoming bounding boxes
  // This process either initializes a new kalman filter instance or run the update measurment function
  // based on an association algorithm that will attempt to match the boxes(t) to boxes(t-1)
  // Unmatched boxes will be instantiated, and matched boxes willbe updated. 

  // Vector to hold the EKF indices that have already been matched to an incoming object measurement
  std::vector<int> matched_detection_indices;
  // Vector to hold array indices of objects to create new EKF instances from
  std::vector<filteredBox> new_detection_indices;



  // Find closest candidate with the highest IoU
  for (auto& bounding_box : bbox_msg->bounding_boxes){
    filteredBox current_box;
    msgBox_to_ekfBox(bounding_box, bbox_msg->header.stamp, current_box);

    filteredBox* current_candidate = nullptr;
    boxEkf* associated_filter = nullptr;
    float IoU_score_current = -1.f;
    float IoU_score_prev = -1.f;
    float IoU_score_max = -1.f;

    for(size_t i=0; i<box_ekfs_.size(); ++i){
      IoU_score_current = IoU(current_box, box_ekfs_[i].getfilteredBox(), IoU_thresh);

      if(IoU_score_current > IoU_score_prev && current_box.darknet_box.Class == box_ekfs_[i].getfilteredBox().darknet_box.Class){
        current_candidate = &current_box;
        associated_filter = &box_ekfs_[i];
        IoU_score_max = IoU_score_current;
      }
      IoU_score_prev = IoU_score_current;
    }
    // If IoU passes and the classes match, we can associate the incoming detection with the existing ekf instance
    if (IoU_score_max >= IoU_thresh){
      associated_filter->updateFilterMeasurement(current_candidate->stamp, *current_candidate);
    }
    else{
      new_detection_indices.push_back(current_box);
    }
  }
  // After trying to associate all incoming object measurements to existing EKF instances,
  // create new EKF instances to track the inputs that weren't associated with existing ones
  for (auto new_object : new_detection_indices) {

    ROS_INFO("Creating new object!");

    box_ekfs_.push_back(boxEkf(new_object));
    // object_ekfs_.back().setQ(cfg_.q_pos, cfg_.q_vel);
    // object_ekfs_.back().setR(cfg_.r_pos);
    }
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

// int yoloEkf::getUniqueId()
//   {
// //return id ++ 
//     int id = 0;
//     bool done = false;
//     while (!done) {
//       done = true;
//       for (auto& track : box_ekfs_) {
//         if (track.getId() == id) {
//           done = false;
//           id++;
//           break;
//         }
//       }
//     }
//     return id;
//   }


}



