#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import threading
import sys
import select
import termios
import tty

class MockSpeechRecognitionNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('mock_speech_recognition', anonymous=True)
        
        # Create publisher for mock speech commands
        self.speech_pub = rospy.Publisher('/mock_speech_recognition/command', String, queue_size=10)
        
        # Define the three commands
        self.start_command = "lets start"
        self.finished_command = "I have finished"
        self.next_object_command = "i need next object"
        
        # Timing parameters
        self.start_delay = rospy.get_param('~start_delay', 1.0)  # seconds
        self.finished_delay = rospy.get_param('~finished_delay', 15.0)  # seconds
        
        # Flag to track if finished command has been sent
        self.finished_sent = False
        
        # Save terminal settings for keyboard input
        self.old_settings = termios.tcgetattr(sys.stdin)
        
        rospy.loginfo("Mock Speech Recognition Node started")
        rospy.loginfo(f"Publishing to: /mock_speech_recognition/command")
        rospy.loginfo(f"Commands:")
        rospy.loginfo(f"  - '{self.start_command}' (after {self.start_delay}s)")
        rospy.loginfo(f"  - '{self.finished_command}' (after {self.finished_delay}s)")
        rospy.loginfo(f"  - '{self.next_object_command}' (press SPACE key)")
        rospy.loginfo("Press 'q' to quit")
        
        # Schedule automatic commands
        self.schedule_automatic_commands()
        
    def schedule_automatic_commands(self):
        """Schedule the automatic commands"""
        # Schedule "lets start" command
        start_timer = threading.Timer(self.start_delay, self.publish_start_command)
        start_timer.daemon = True
        start_timer.start()
        
        # Schedule "I have finished" command
        finished_timer = threading.Timer(self.finished_delay, self.publish_finished_command)
        finished_timer.daemon = True
        finished_timer.start()
    
    def publish_command(self, command):
        """Publish a speech command"""
        msg = String()
        msg.data = command
        self.speech_pub.publish(msg)
        rospy.loginfo(f"Published command: '{command}'")
    
    def publish_start_command(self):
        """Publish the start command"""
        if not rospy.is_shutdown():
            self.publish_command(self.start_command)
    
    def publish_finished_command(self):
        """Publish the finished command"""
        if not rospy.is_shutdown() and not self.finished_sent:
            self.publish_command(self.finished_command)
            self.finished_sent = True
    
    def publish_next_object_command(self):
        """Publish the next object command"""
        if not rospy.is_shutdown():
            self.publish_command(self.next_object_command)
    
    def get_key(self):
        """Get keyboard input (non-blocking)"""
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        return key
    
    def run(self):
        """Main loop - monitor keyboard input"""
        try:
            tty.setcbreak(sys.stdin.fileno())
            
            while not rospy.is_shutdown():
                key = self.get_key()
                
                if key == ' ':  # Space key
                    self.publish_next_object_command()
                elif key == 'q' or key == '\x03':  # 'q' or Ctrl+C
                    rospy.loginfo("Quitting...")
                    break
                
                # Small sleep to prevent CPU spinning
                rospy.sleep(0.01)
                
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

def main():
    try:
        node = MockSpeechRecognitionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupt received")
    finally:
        # Make sure terminal settings are restored
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, termios.tcgetattr(sys.stdin))

if __name__ == '__main__':
    main()