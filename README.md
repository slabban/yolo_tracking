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

You should be good to go now! You can use this system on a live camera or rosbag file as long as you configure your darknet_ros package to subscribe to the respective camera topics.

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

Association between the latest detections and the existing EKF tracked boxes is done by comparing the Intersection Over Union between the two. If the final IOU value is above our specified threshold of 0.3, then the detection can be matched to the exisiting EKF, otherwise, and new EKF object will be instantiated for that detection. 

The second part of *yolo_ekf* contains a timer callback that is used to make predictions on the tracked ekf boxes:

1. Estimation
2. Track Identity Destruction

The timer callback rate is completely dependent on the rate of detections of the YOLO model being run. While using YOLOV3, I was getting an detection output rate of roughly 17 HZ, and was able to push the estimate rate to 50 HZ (0.02 seconds). The YOLOV4 model architecture is larger, which really impacted the detection rate, bringing it down to ~7.5 HZ, after some experimentation I found that a estimate rate of 25 HZ (0.04 seconds) to be fine number.

## Tuning the EKF and Association Algorithm Parameters
Now for the fun part, tuning.. Sarcasm aside, this is arguably the most important part of getting the desired performance from the system. To make life a little easier, I've added a dynamic reconfigure feature to this package. Tuned to the following numbers:

<img src="/images/yolo_ekf_rqt.png?raw=true" />

Where the parameters

| Parameter | Description |
| ----------- | ----------- |
| IoU_thresh | Maximum distance to associate measurement |
| min_width | Minimum bounding box width for tracking | 
| min_height | Minimum bounding box width for tracking | 
| max_age | Max Age to delete ekf instance | 
| min_age | Minimum Age to publish EKF instance | 
| p_factor | Inital factor to set P diagonal to |
| r_cx_cy | EKF state standard deviation squared |
| r_w_h | EKF state standard deviation squared |
| q_pos | EKF measurement standard deviation squared |
| q_vel | EKF measurement standard deviation squared |

## Performance Analysis 
I tested the performance of the algorithm on a bag file I was able to access from an archived Udacity course for a visual analysis:

*insert video here*

Due to the constraints of my current system, I had to step the playback speed down, but the general performance can be seen here. I suspect that the system, specifically the EKF Process Noise could use some more tuning through the introduction of cross-covariances.

The current methodolgy has a few drawbacks that apply to these classical systems. Namely loss of tracking through ID Swaps and Object Occlusion. Using the famous Hungarian Algorithm for association paired with the EKF is known as **SORT**, and would help reduce the frequency of ID swaps and improve the overall effciency, since that algorithm uses a linear assignment strategy; whereas my current approach associates using a 'first come first serve' method. But the best method to date is called **Deepsort**, which leverages feature maps from the CNN for association. Deepsort elegantly solves the ID Swap and Occlusion problem, and is built on **SORT** and the classical concepts that were used in this repository. You can get the full scoop on this [here](https://medium.com/augmented-startups/deepsort-deep-learning-applied-to-object-tracking-924f59f99104).



## Potential Future Tasks
I don't quite intend to revisit this repository to build it further, but there's no harm in identifying a couple of next steps that can be taken.

- [ ] Refine Process Noise tuning 
- [ ] Add Markers to vizualize the estimated velocities
- [ ] Containerize the System
- [ ] Port the System to ROS2

   



















