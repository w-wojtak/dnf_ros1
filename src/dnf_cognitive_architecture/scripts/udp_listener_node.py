#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import socket
import threading

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

class UDPListenerNode(object):
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('udp_listener_node', anonymous=True)
        self.filtered_publisher = rospy.Publisher(
            'voice_command_filtered', String, queue_size=10)

        # Track "start" and "finished" to publish only once
        self.sent_flags = {
            "start": False,
            "finished": False
        }

        rospy.loginfo('Listening for UDP messages on port %d', UDP_PORT)
        self.thread = threading.Thread(target=self.listen_loop)
        self.thread.daemon = True
        self.thread.start()

    def filter_message(self, message):
        """
        Returns message if it should be forwarded, otherwise None.
        """
        message = message.lower().strip()
        if message in self.sent_flags:
            if not self.sent_flags[message]:
                self.sent_flags[message] = True
                return message
            else:
                return None  # Already sent once, skip
        return message  # Other commands always forwarded

    def listen_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        while not rospy.is_shutdown():
            try:
                data, _ = sock.recvfrom(1024)
                message = data.decode().strip()
                rospy.loginfo("Received: '%s'", message)

                filtered = self.filter_message(message)
                if filtered:
                    self.filtered_publisher.publish(String(data=filtered))
                    rospy.loginfo("Published filtered message: '%s'", filtered)
                else:
                    rospy.loginfo("Filtered out (duplicate or unwanted).")
            except Exception as e:
                rospy.logerr("Error in UDP listener: %s", str(e))

def main():
    node = UDPListenerNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()