#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import random
import time

class MockSpeechRecognitionNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('mock_speech_recognition', anonymous=True)
        
        # Create publisher for mock speech commands
        self.speech_pub = rospy.Publisher('/mock_speech_recognition/command', String, queue_size=10)
        
        # Define available commands - easy to extend by adding more entries
        self.commands = [
            "pick up the blue box",
            "move to the base",
            "stop operation"
        ]
        
        # Optional: Add command variations for more realistic simulation
        self.command_variations = {
            "pick up the blue box": ["pick up blue box", "grab the blue box", "get blue box"],
            "move to the base": ["go to base", "return to base", "move to base position"],
            "stop operation": ["stop", "halt", "emergency stop"]
        }
        
        # Parameters
        self.publish_rate = rospy.get_param('~publish_rate', 0.5)  # Hz
        self.random_mode = rospy.get_param('~random_mode', True)
        self.command_index = 0
        
        rospy.loginfo("Mock Speech Recognition Node started")
        rospy.loginfo(f"Publishing to: /mock_speech_recognition/command")
        rospy.loginfo(f"Available commands: {self.commands}")
        rospy.loginfo(f"Publishing rate: {self.publish_rate} Hz")
        rospy.loginfo(f"Random mode: {self.random_mode}")
        
    def get_next_command(self):
        """Get the next command to publish"""
        if self.random_mode:
            # Random selection from available commands
            base_command = random.choice(self.commands)
            
            # Optionally use a variation of the command
            if base_command in self.command_variations and random.random() < 0.3:
                return random.choice(self.command_variations[base_command])
            else:
                return base_command
        else:
            # Sequential mode - cycle through commands
            command = self.commands[self.command_index]
            self.command_index = (self.command_index + 1) % len(self.commands)
            return command
    
    def publish_command(self, command):
        """Publish a speech command"""
        msg = String()
        msg.data = command
        self.speech_pub.publish(msg)
        rospy.loginfo(f"Published command: '{command}'")
    
    def run(self):
        """Main loop"""
        rate = rospy.Rate(self.publish_rate)
        
        while not rospy.is_shutdown():
            # Get and publish next command
            command = self.get_next_command()
            self.publish_command(command)
            
            # Sleep to maintain publish rate
            rate.sleep()

def main():
    try:
        node = MockSpeechRecognitionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()