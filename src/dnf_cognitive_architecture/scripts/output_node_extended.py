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
            if -45 <= self.received_value <= -35:
                print("Message: Threshold crossed near the left input position.")
            elif -5 <= self.received_value <= 5:
                print("Message: Threshold crossed near the center input position.")
            elif 35 <= self.received_value <= 45:
                print("Message: Threshold crossed near the right input position.")
            else:
                print(
                    "Message: Threshold crossing detected outside expected input positions.")

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
