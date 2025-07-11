
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


Terminal 3:
cd ~/dnf_ros1
source venv/bin/activate
cd src/dnf_cognitive_architecture/scripts
python object_detector.py


"""


"""
table corners:
upper left: x = -0.82, y = 0.06
lower left: x = -0.82, y = 0.52
upper right: x = 0.71, y = 0.06
lower right: x = 0.71, y = 0.52

"""


"""

    Create 4 input matrices (one for each object class: base, load, bearing, motor)
    Map each object to a specific gaussian position: base→-60, load→-20, bearing→20, motor→40
    Trigger a gaussian input when an object moves more than 0.05 in x or y
    Each trigger creates a 1-second duration gaussian starting at the current time

    
"""