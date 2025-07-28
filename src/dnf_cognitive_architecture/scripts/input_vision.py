#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray, String
import json


class InputMatrix(object):
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('input_matrix', anonymous=True)

        # Create a publisher for matrices in a combined format
        self.input_pub = rospy.Publisher(
            'input_matrices_combined', Float32MultiArray, queue_size=10)
        
        # Create a subscriber for object detections
        self.detection_sub = rospy.Subscriber(
            '/object_detections',
            String,
            self.detection_callback,
            queue_size=10
        )

        # Timer to publish every full time step
        self.timer = rospy.Timer(rospy.Duration(1.0), self.publish_slices)

        # Simulation parameters
        self.x_lim = 80
        self.t_lim = 15
        self.dx = 0.2
        self.dt = 0.1

        # Object class to gaussian position mapping
        self.object_positions = {
            'base': -60,
            'load': -20,
            'bearing': 20,
            'motor': 40
        }
        
        # Store last known positions for movement detection
        self.last_positions = {
            'base': None,
            'load': None,
            'bearing': None,
            'motor': None
        }
        
        # Movement threshold
        self.movement_threshold = 0.05
        
        # Gaussian parameters
        self.amplitude = 5.0
        self.width = 2.0
        self.duration = 1.0  # 1 second duration

        # Initialize lists to store active gaussians for both matrices
        self.active_gaussians_matrix1 = []
        self.active_gaussians_matrix2 = []

        # Define spatial and temporal grids
        self.x = np.arange(-self.x_lim, self.x_lim + self.dx, self.dx)
        self.t = np.arange(0, self.t_lim + self.dt, self.dt)

        # Initialize input matrices
        self.input_matrix1 = np.zeros((len(self.t), len(self.x)))
        self.input_matrix2 = np.zeros((len(self.t), len(self.x)))
        self.input_matrix_3 = np.zeros((len(self.t), len(self.x)))

        # Initialize the current time index for publishing
        self.current_time_index = 0

        # for tracking detected movements
        self.movement_detected = set()

    def gaussian(self, center=0, amplitude=1.0, width=1.0):
        return amplitude * np.exp(-((self.x - center) ** 2) / (2 * (width ** 2)))

    def check_movement(self, object_name, new_x, new_y):
        """Check if object has moved beyond threshold"""
        if self.last_positions[object_name] is None:
            self.last_positions[object_name] = {'x': new_x, 'y': new_y}
            return False
        
        last_x = self.last_positions[object_name]['x']
        last_y = self.last_positions[object_name]['y']
        
        dx = abs(new_x - last_x)
        dy = abs(new_y - last_y)
        
        if dx > self.movement_threshold or dy > self.movement_threshold:
            self.last_positions[object_name] = {'x': new_x, 'y': new_y}
            return True
        
        return False

    def add_gaussian_input(self, object_name):
        """Add a gaussian input for the specified object to both matrices"""
        current_time = self.t[self.current_time_index]
        t_start = current_time
        t_stop = t_start + self.duration
        
        # Get the gaussian position for this object
        center = self.object_positions[object_name]
        
        # Create gaussian parameters
        gaussian_params = {
            'center': center,
            'amplitude': self.amplitude,
            'width': self.width,
            't_start': t_start,
            't_stop': t_stop
        }
        
        # Add to both matrices' active gaussians lists
        self.active_gaussians_matrix1.append(gaussian_params.copy())
        self.active_gaussians_matrix2.append(gaussian_params.copy())
        
        rospy.loginfo(f"Added gaussian input for {object_name} at position {center} starting at t={t_start:.2f}")

    def update_input_matrices(self):
        """Update both input matrices based on their active gaussians"""
        current_time = self.t[self.current_time_index]
        
        # Update matrix 1
        self.input_matrix1[self.current_time_index] = np.zeros(len(self.x))
        active_gaussians_copy1 = self.active_gaussians_matrix1.copy()
        
        for gaussian in active_gaussians_copy1:
            if gaussian['t_start'] <= current_time <= gaussian['t_stop']:
                self.input_matrix1[self.current_time_index] += self.gaussian(
                    center=gaussian['center'],
                    amplitude=gaussian['amplitude'],
                    width=gaussian['width']
                )
            elif current_time > gaussian['t_stop']:
                self.active_gaussians_matrix1.remove(gaussian)

        # Update matrix 2
        self.input_matrix2[self.current_time_index] = np.zeros(len(self.x))
        active_gaussians_copy2 = self.active_gaussians_matrix2.copy()
        
        for gaussian in active_gaussians_copy2:
            if gaussian['t_start'] <= current_time <= gaussian['t_stop']:
                self.input_matrix2[self.current_time_index] += self.gaussian(
                    center=gaussian['center'],
                    amplitude=gaussian['amplitude'],
                    width=gaussian['width']
                )
            elif current_time > gaussian['t_stop']:
                self.active_gaussians_matrix2.remove(gaussian)

    def detection_callback(self, msg):
        """Callback for object detection messages"""
        try:
            detection_data = json.loads(msg.data)
            detections = detection_data.get('detections', [])
            
            for detection in detections:
                object_name = detection.get('object', 'Unknown')
                position = detection.get('position', {})
                
                x = position.get('x', 0.0)
                y = position.get('y', 0.0)
                
                # Check if this is one of our tracked objects
                if object_name in self.object_positions:
                    # Only process if not already detected
                    if object_name not in self.movement_detected:
                        if self.check_movement(object_name, x, y):
                            rospy.loginfo(f"Movement detected for {object_name}")
                            self.add_gaussian_input(object_name)
                            self.movement_detected.add(object_name)
            
            # Update input matrices with all active gaussians
            self.update_input_matrices()
            
        except Exception as e:
            rospy.logerr(f"Error processing detection message: {e}")

    def publish_slices(self, event):
        if self.current_time_index < len(self.t):
            # Update input matrices with active gaussians
            self.update_input_matrices()
            
            # Create combined input (matrix 1, matrix 2, and matrix 3)
            combined_input = [
                self.input_matrix1[self.current_time_index].tolist(),
                self.input_matrix2[self.current_time_index].tolist(),
                self.input_matrix_3[self.current_time_index].tolist()
            ]

            # Create and publish message
            msg = Float32MultiArray()
            msg.data = [item for sublist in combined_input for item in sublist]
            self.input_pub.publish(msg)

            # Log publication
            rospy.loginfo(
                f"Published t={self.t[self.current_time_index]:.2f}, "
                f"Max matrix1: {self.input_matrix1[self.current_time_index].max():.2f}, "
                f"Max matrix2: {self.input_matrix2[self.current_time_index].max():.2f}"
            )

            self.current_time_index += 1
        else:
            rospy.loginfo("Completed publishing all time slices.")
            self.timer.shutdown()

def main():
    try:
        node = InputMatrix()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()

