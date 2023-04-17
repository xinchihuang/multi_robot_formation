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
* Set color threshold
 
  Flow the instructions in [cmvision_3d](http://wiki.ros.org/cmvision_3d) to set robot's color accordingly
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
* Robot connection
  1. Make sure computer and all robots are connected to same WIFI(Follow the [instruction](https://robomaster-dev.readthedocs.io/en/latest/) to connect robomaster)
  2. Run check_ip.py to get each robots' IP (connect robot to WIFI one by one )
  3. Record the IP address of each robot by setting the self.IP_DICT attribute using ImageListener in expert_control.py. Ensure that the IP address corresponds to the color of each robot.
* Expert Control
  1. Start ROS
  
  `rosecore`
  
  2. Start realsense camera
  
  `roslaunch realsense2_camera rs_rgbd.launch `
  
  3. Start cmvision_3d
  
  `roslaunch cmvision_3d color_tracker.launch `
  
  4. Start expert control 
  
  `rosrun multi_robot_formation expert_control.py `
  
### Interactions with other models
    
  Interface is defined in model_control_ros.py. You can use the ModelControl as ros interface or you can define your own.

