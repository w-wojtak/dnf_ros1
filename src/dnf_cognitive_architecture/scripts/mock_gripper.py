#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import threading

class MockRobotGripper:
    def __init__(self):
        rospy.init_node('mock_robot_gripper', anonymous=True)
        self.delay = rospy.get_param('~process_time', 10.0)  # seconds

        self.status_pub = rospy.Publisher('/gripper/status', String, queue_size=10)
        self.command_sub = rospy.Subscriber('/gripper/command', String, self.command_callback)

        rospy.loginfo("Mock Robot Gripper node started. Waiting for commands...")

    def command_callback(self, msg):
        object_name = msg.data
        rospy.loginfo("Received gripper command for object: %s", object_name)
        # Start a timer thread to simulate the delay
        threading.Thread(target=self.process_object, args=(object_name,)).start()

    def process_object(self, object_name):
        rospy.loginfo("Processing object '%s' for %.1f seconds...", object_name, self.delay)
        rospy.sleep(self.delay)
        confirmation_msg = String()
        confirmation_msg.data = object_name
        self.status_pub.publish(confirmation_msg)
        rospy.loginfo("Published gripper status for object: %s", object_name)

if __name__ == '__main__':
    try:
        node = MockRobotGripper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass