#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32MultiArray


class OutputNode(object):
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('output_node', anonymous=True)

        # Subscription to the topic publishing threshold-crossing x values
        self.subscription = rospy.Subscriber(
            'threshold_crossings',  # Make sure this matches the topic name in the publisher node
            Float32MultiArray,
            self.listener_callback,
            queue_size=10)

        # Timer that triggers periodically to print based on received values
        self.timer = rospy.Timer(
            rospy.Duration(1.0), self.timer_callback)  # Runs every 1 second

        # Initialize received value to None
        self.received_value = None

    def listener_callback(self, msg):
        if msg.data:
            self.received_value = msg.data[0]
            rospy.loginfo(
                "Received threshold crossing value: {:.2f}".format(self.received_value))

        if self.received_value is not None:
            if -65 <= self.received_value <= -55:
                rospy.loginfo("HAND OVER BASE")
            elif -25 <= self.received_value <= -15:
                rospy.loginfo("HAND OVER LOAD")
            elif 15 <= self.received_value <= 25:
                rospy.loginfo("HAND OVER BEARING")
            elif 35 <= self.received_value <= 45:
                rospy.loginfo("HAND OVER MOTOR")
            else:
                rospy.loginfo("Message: Threshold crossing detected outside expected input positions.")

        else:
            rospy.loginfo("Received message with empty data.")

    def timer_callback(self, event):
        pass


def main():
    try:
        output_node = OutputNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
