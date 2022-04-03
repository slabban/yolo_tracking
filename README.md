# Multiple Object Tracking of YOLO Detections Using Kalman Filtration in ROS
My Implementation of Computer Vision Multi-Object Tracking using an Extended Kalman Filter (EKF) and an Intersection Over Union (IOU) association algorithm within the Robot Operating System (ROS) middleware.

The detection algorithm uses the YOLO Darknet ROS framework. The frameworks tested so far are from the original [leggedrobotics darknet-ros YOLOV3](https://github.com/leggedrobotics/darknet_ros) implementation and the [yolov4-for-darknet_ros](https://github.com/Tossy0423/yolov4-for-darknet_ros) extended implementation by Tossy0423.

The tuning parameters, which I will get into later in this read me, are optimized for the latter YOLOv4 implementation using my current setup, so you will likely need to tune the parameters to better fit your machine's processing capability.


## System Configuration
- ROS Melodic
- Ubuntu 18.04
- CUDA version 10.2
- Intel Core i7-6700HQ CPU @ 2.60GHz
- NVIDIA GeForce GTX 1060
- NVIDIA Driver Version 470.63.01



## Installation
The first step is to set up workspace and darknet_ros repository, so please make sure to pick [leggedrobotics darknet-ros YOLOV3](https://github.com/leggedrobotics/darknet_ros) or [yolov4-for-darknet_ros](https://github.com/Tossy0423/yolov4-for-darknet_ros) and closesly follow their respective instructions. This part can have a learning curve depending on your luck with your NVIDIA GPU. I also recommend you test it with a webcam to make sure all is working as expected.

This repository should be cloned under the **src** directory of the same workspace that your chosen darknet_ros is also cloned under. Once that is done you can install any additional dependencies using:
>`rosdep install --from-paths src --ignore-src -r -y`

After that you can build the system using catkin build or catkin_make. I used the catkin build option and built in release mode for performance:
>`catkin build -DCMAKE_BUILD_TYPE=Release`

You should be good to go now!

## Methodology
The problem solved is Multi-Object tracking (MOT) of the array of bounding boxes from the YOLO CNN detections, which is acheived using a combination of the classical but effective Extended Kalman Filter along with an association algorithm to manage multiple detections. 

The *yolo_ekf* package is built from two main components:
- *box_ekf*
- *yolo_ekf* - this is a ros node, it shares the same name as the package

*box_ekf* is the class implementation of the Extended Kalman Filter where the State Vector, Covariance Matrix, Transition Matrices, Process Noise Covariance, Measurement Noise Covariance are all defined. Along with the necessary Update Prediction and Update Measurements. My source of knowldege for this specific EKF strategy, and my inspiration to take on this project came from this [article](https://thinkautonomous.medium.com/computer-vision-for-tracking-8220759eee85that) Jeremy Cohen had written.

The *yolo_ekf* ros node implementation that comprises of a subscriber that synchronizes the darknet bounding boxes+sensor_msgs/Image topic. In the callback for this subscriber, we tackle:

1. Detection
2. Association
3. Track Identity Creation

Association between the latest detections and the existing EKF tracked boxes is done by comparing the Intersection Over Union between the two. If the final IOU value is above our specified threshold of 0.33, then the detection can be matched to the exisiting EKF, otherwise, and new EKF object will be instantiated for that detection. 

The second part of *yolo_ekf* contains a timer callback that is used to make predictions on the tracked ekf boxes:

1. Estimation
2. Track Identity Destruction

















