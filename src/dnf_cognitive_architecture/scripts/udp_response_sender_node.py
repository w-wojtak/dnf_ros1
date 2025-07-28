#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import socket

# Change this to the IP of your Windows machine
UDP_IP = "10.205.240.222"   # example Windows IP (adjust!)
UDP_PORT = 5006             # different port than the listener, to avoid conflict

class UDPResponseSenderNode(object):
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('udp_response_sender_node', anonymous=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.subscriber = rospy.Subscriber(
            'response_command',
            String,
            self.listener_callback,
            queue_size=10
        )
        rospy.loginfo("UDP Response Sender ready. Sending to %s:%d", UDP_IP, UDP_PORT)

    def listener_callback(self, msg):
        message = msg.data.strip()
        try:
            self.sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
            rospy.loginfo("Sent response: '%s'", message)
        except Exception as e:
            rospy.logerr("Failed to send UDP response: %s", str(e))

def main():
    node = UDPResponseSenderNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()