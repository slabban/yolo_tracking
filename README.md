# yolo_tracking
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
`rosdep install --from-paths src --ignore-src -r -y`

After that you can build the system, and you should be good to go!



## Methodology
TODO












