
# source /opt/ros/noetic/setup.bash
# source ~/dnf_ros1/devel/setup.bash

"""
cd ~/dnf_ros1
catkin_make

Terminal 1:

source ~/dnf_ros1/devel/setup.bash
roscore

Terminal 2:

source ~/dnf_ros1/devel/setup.bash
rosrun dnf_cognitive_architecture dnf_model_learning.py

from launch file:
roslaunch dnf_cognitive_architecture dnf_learn.launch


to run vision code:
source venv_detection/bin/activate


"""