cmake_minimum_required(VERSION 3.0.2)
project(dnf_cognitive_architecture_extended)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_python_setup()

catkin_package()

install(
  PROGRAMS
    scripts/dnf_model_learning.py
    scripts/dnf_model_recall.py
    scripts/input_matrix.py
    scripts/output_node.py
    scripts/rabbit_subscriber_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY dnf_cognitive_architecture_extended/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
  FILES_MATCHING PATTERN "*.py"
)
