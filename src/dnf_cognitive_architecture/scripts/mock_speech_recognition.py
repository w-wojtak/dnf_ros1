#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import sys
import select
import termios
import tty

KEY_COMMANDS = {
    's': 'start',
    'f': 'finished',
    'n': 'next',
}

def get_key():
    """Get a single keypress (non-blocking, no Enter needed)"""
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    key = ''
    if rlist:
        key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    return key

if __name__ == '__main__':
    rospy.init_node('mock_speech_recognition', anonymous=True)
    pub = rospy.Publisher('/mock_speech_recognition/command', String, queue_size=10)
    old_settings = termios.tcgetattr(sys.stdin)

    rospy.loginfo("Press: s=start, f=finished, n=next, q=quit")

    try:
        while not rospy.is_shutdown():
            key = get_key()
            if key in KEY_COMMANDS:
                msg = KEY_COMMANDS[key]
                pub.publish(msg)
                rospy.loginfo("Published: '%s'" % msg)
            elif key == 'q':
                rospy.loginfo("Quitting...")
                break
            rospy.sleep(0.05)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)