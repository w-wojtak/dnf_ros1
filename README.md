# DNF Cognitive Architecture with Object Detection

This repository contains a cognitive architecture based on Dynamic Neural Fields (DNF) with real-time object detection using ZED camera and YOLO.

## System Overview

The system consists of:

1. **Object Detection** - ZED camera + YOLO object detection (runs in Python virtual environment)
2. **DNF Learning** - Dynamic Neural Field model for learning sequences
3. **DNF Recall** - Dynamic Neural Field model for recalling learned sequences
4. **Input/Output Processing** - Nodes for handling inputs and outputs

## Prerequisites

- Ubuntu 20.04
- ROS Noetic
- Python 3.8+
- CUDA-capable GPU
- ZED Camera and SDK
- PyTorch


## Setup

### Create virtual environment for object detection
```
cd ~/dnf_ros1
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies in venv
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python numpy matplotlib
pip install rospkg catkin_pkg pyyaml
python /usr/local/zed/get_python_api.py
```

### Create ROS symlinks in venv
```
ln -s /opt/ros/noetic/lib/python3/dist-packages/rospy ~/dnf_ros1/venv/lib/python3.8/site-packages/
ln -s /opt/ros/noetic/lib/python3/dist-packages/std_msgs ~/dnf_ros1/venv/lib/python3.8/site-packages/
ln -s /opt/ros/noetic/lib/python3/dist-packages/rosgraph ~/dnf_ros1/venv/lib/python3.8/site-packages/
ln -s /opt/ros/noetic/lib/python3/dist-packages/roslib ~/dnf_ros1/venv/lib/python3.8/site-packages/
ln -s /opt/ros/noetic/lib/python3/dist-packages/geometry_msgs ~/dnf_ros1/venv/lib/python3.8/site-packages/
```

## Running the System
### Terminal 1: ROS Core
```
roscore
```

### Terminal 2: Object Detection (with venv)
```
cd ~/dnf_ros1
source venv/bin/activate
cd src/dnf_cognitive_architecture/scripts
python object_detector.py

# Controls:
# 's' - Start detection
# 'p' - Pause detection
# 'q' - Quit
```

### Terminal 3: DNF Learning
```
cd ~/dnf_ros1
source devel/setup.bash
rosrun dnf_cognitive_architecture dnf_model_learning.py
```

### Terminal 4: Monitor Detections (optional)
```
rostopic echo /object_detections
```