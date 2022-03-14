// ROS and node class header file
#include <ros/ros.h>
#include "yolo_ekf.hpp"

int main(int argc, char** argv)
{
  // Initialize ROS and declare node handles
  ros::init(argc, argv, "yolo_ekf_node");
  ros::NodeHandle n;
  ros::NodeHandle pn("~");

  // Instantiate node class
  yolo_ekf::yoloEkf node(n, pn);

  // Spin and process callbacks
  ros::spin();
}