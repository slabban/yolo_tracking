cmake_minimum_required(VERSION 3.0.2)
project(yolo_ekf)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  dynamic_reconfigure
  image_transport
  image_geometry
  cv_bridge
  darknet_ros_msgs
  darknet_ros 
  message_generation
  message_filters
)

find_package(OpenCV REQUIRED)

# generate_dynamic_reconfigure_options(
#   cfg/CameraLidarFusion.cfg
#   cfg/Lidar_ekf.cfg
# )



# generate_messages(
#   DEPENDENCIES
#   std_msgs
# )

catkin_package(
  DEPENDS OpenCV std_msgs roscpp darknet_ros_msgs 
)



include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}
 src/yolo_ekf.cpp
 src/yolo_ekf_node.cpp
 src/box_ekf.cpp

)

add_dependencies(${PROJECT_NAME}
  # ${PROJECT_NAME}_gencfg
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)


install(
  TARGETS
    ${PROJECT_NAME}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)