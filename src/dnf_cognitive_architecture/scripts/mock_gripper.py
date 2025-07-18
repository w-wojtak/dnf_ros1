#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Bool
import time
import random

class MockRobotGripperNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('mock_robot_gripper', anonymous=True)
        
        # Create publisher for acknowledging next object requests
        self.ack_pub = rospy.Publisher('/next_object_acknowledge', Bool, queue_size=10)
        
        # Subscribe to speech commands to detect next object requests
        self.speech_sub = rospy.Subscriber('/mock_speech_recognition/command', String, self.speech_callback)
        
        # Parameters
        self.min_process_time = rospy.get_param('~min_process_time', 2.0)  # Min seconds to process
        self.max_process_time = rospy.get_param('~max_process_time', 5.0)  # Max seconds to process
        
        # Tracking if we're currently processing a request
        self.processing = False
        
        rospy.loginfo("Mock Robot Gripper Node started")
        rospy.loginfo("Listening for 'next object' requests and sending acknowledgements")
        
    def speech_callback(self, msg):
        """Handle speech commands"""
        command = msg.data.lower()
        
        if ("i need next object" in command or "next object" in command) and not self.processing:
            self.processing = True
            rospy.loginfo("GRIPPER: Detected next object request, starting to process...")
            
            # Process the request in a separate thread
            import threading
            process_thread = threading.Thread(target=self.process_request)
            process_thread.daemon = True
            process_thread.start()
    
    def process_request(self):
        """Simulate processing time for robot to handle the object"""
        try:
            # Random processing time
            process_time = random.uniform(self.min_process_time, self.max_process_time)
            
            # Simulate robot moving to next object
            rospy.loginfo(f"GRIPPER: Moving to next object (will take {process_time:.1f} seconds)...")
            
            # Sleep to simulate processing time
            rospy.sleep(process_time/2)
            rospy.loginfo("GRIPPER: Reaching object position...")
            rospy.sleep(process_time/2)
            
            # Send acknowledgement
            ack_msg = Bool()
            ack_msg.data = True
            self.ack_pub.publish(ack_msg)
            rospy.loginfo("GRIPPER: Object handling complete, sent acknowledgement")
            
            # Reset processing flag
            self.processing = False
            
        except Exception as e:
            rospy.logerr(f"Error in process_request: {str(e)}")
            self.processing = False

def main():
    try:
        gripper = MockRobotGripperNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()