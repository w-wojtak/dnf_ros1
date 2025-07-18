cmake_minimum_required(VERSION 3.0.2)
project(dnf_cognitive_architecture)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_python_setup()

catkin_package()

install(
  PROGRAMS
    scripts/dnf_model_learning.py
    scripts/dnf_model_learning_speech.py
    scripts/dnf_model_recall.py
    scripts/dnf_model_learning_extended.py
    scripts/dnf_model_recall_extended.py
    scripts/input_matrix.py
    scripts/input_vision.py
    scripts/output_node.py
    scripts/output_node_extended.py
    scripts/object_detector.py
    scripts/object_detector_table.py
    scripts/mock_speech_recognition.py
    scripts/mock_gripper.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY dnf_cognitive_architecture/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
  FILES_MATCHING PATTERN "*.py"
)
