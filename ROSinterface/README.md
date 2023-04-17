# A simple platform for multi-robot control with ROS
## Prerequisites
* Robot platform: Robomaster EP
* Sensor: Intel realsense D435i
* Operating system: Ubuntu 18.04/Ubuntu 20.04
* ROS version: melodic/noetic 
## Dependencies
* ROS package
  * [cmvision](http://wiki.ros.org/cmvision)
  * [cmvision_3d](http://wiki.ros.org/cmvision_3d)
* Intel® RealSense™ SDK 2.0 
* Opencv 3.0
* Other python package
  * pyrealsense2
  * cv2
  * socket
## Usage
### Localization
* Calibration
  1. Set robot's color in pose_recorder.py by adjusting parameter calibration_color
  2. Put the robot for calibration on the ground and record its position
  3. Run pose_recorder.py follow the instructions type in the robots position in real world
  4. Repeat step two and three to get more data for calibration (at least 4 times)
  5. Run calibration.py to get extrinsic matrix which is stored at params.csv
* Localization
  1. Set robot's color in localization.py
  2. Run localization.py to get robot's position
### Multi-robot formation with expert control
### Interactions with other models

