#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray


class InputMatrix(object):

    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('input_matrix', anonymous=True)

        # Create a publisher for both input matrices in a combined format
        self.input_pub = rospy.Publisher(
            'input_matrices_combined', Float32MultiArray, queue_size=10)

        # Create a subscriber for the threshold crossings topic
        self.threshold_sub = rospy.Subscriber(
            'threshold_crossings',
            Float32MultiArray,
            self.threshold_callback,
            queue_size=10
        )

        # Timer to publish every full time step
        self.timer = rospy.Timer(rospy.Duration(1.0), self.publish_slices)

        # Simulation parameters
        self.x_lim = 80
        self.t_lim = 15
        self.dx = 0.2
        self.dt = 0.1

        # Define input parameters for matrix 1
        input_position = [-40, 0, 40]
        amplitude = 5.0
        width = 2.0
        t_start_list = [1, 4, 7]
        t_stop_list = [2, 5, 8]

        if len(input_position) != len(t_start_list) or len(input_position) != len(t_stop_list):
            raise ValueError(
                "input_position, t_start_list, and t_stop_list must have the same length.")

        # Define Gaussian parameters for matrix 1
        self.gaussian_params_1 = [
            {'center': pos, 'amplitude': amplitude, 'width': width,
             't_start': t_start, 't_stop': t_stop}
            for pos, t_start, t_stop in zip(input_position, t_start_list, t_stop_list)
        ]

        t_start_list_2 = [1.5, 5, 7.5]
        t_stop_list_2 = [2.5, 6, 8.5]

        self.gaussian_params_2 = [
            {'center': pos, 'amplitude': amplitude, 'width': width,
             't_start': t_start, 't_stop': t_stop}
            for pos, t_start, t_stop in zip(input_position, t_start_list_2, t_stop_list_2)
        ]

        # Initialize Gaussian parameters for matrix 2 (empty initially)
        self.gaussian_params_3 = []

        # Define spatial and temporal grids
        self.x = np.arange(-self.x_lim, self.x_lim + self.dx, self.dx)
        self.t = np.arange(0, self.t_lim + self.dt, self.dt)

        # Generate the input matrix for Gaussian parameters 1
        self.input_matrix_1 = self.get_input_matrix(self.gaussian_params_1)
        self.input_matrix_2 = self.get_input_matrix(self.gaussian_params_2)
        self.input_matrix_3 = np.zeros(
            (len(self.t), len(self.x)))  # Placeholder

        # Initialize the current time index for publishing
        self.current_time_index = 0

    def gaussian(self, center=0, amplitude=1.0, width=1.0):
        return amplitude * np.exp(-((self.x - center) ** 2) / (2 * (width ** 2)))

    def get_input_matrix(self, params_list):
        input_matrix = np.zeros((len(self.t), len(self.x)))
        for params in params_list:
            center = params['center']
            amplitude = params['amplitude']
            width = params['width']
            t_start = params['t_start']
            t_stop = params['t_stop']

            for i, t_val in enumerate(self.t):
                if t_start <= t_val <= t_stop:
                    input_matrix[i,
                                 :] += self.gaussian(center, amplitude, width)
        return input_matrix

    def threshold_callback(self, msg):
        # Extract the threshold crossing value
        self.received_value = msg.data[0]
        rospy.loginfo(
            "Received threshold crossing value: {:.2f}".format(
                self.received_value)
        )

        # Determine the input position based on the received threshold crossing value
        if -45 <= self.received_value <= -35:
            input_position = -40  # Left input position
        elif -5 <= self.received_value <= 5:
            input_position = 0  # Center input position
        elif 35 <= self.received_value <= 45:
            input_position = 40  # Right input position
        else:
            rospy.logwarn(
                "Threshold crossing value is outside expected ranges.")
            return  # Exit if the value is not within valid ranges

        # Define the delay in time steps
        delay = 5

        # Calculate t_start and t_stop for the updated Gaussian input
        t_start = self.t[self.current_time_index] + delay * self.dt
        t_stop = t_start + (self.dt * 10)  # Keep the duration of input fixed

        # Update gaussian_params_2 with new values
        self.gaussian_params_3 = [
            {'center': input_position, 'amplitude': 5.0, 'width': 2.0,
             't_start': t_start, 't_stop': t_stop}
        ]

        # Regenerate input_matrix_2 with updated parameters
        self.input_matrix_3 = self.get_input_matrix(self.gaussian_params_3)

        rospy.loginfo(
            "Updated gaussian_params_3: {}".format(self.gaussian_params_3)
        )

    def publish_slices(self, event):
        if self.current_time_index < len(self.t):
            # Combine the two matrices into a single array (nested)
            combined_input = [
                self.input_matrix_1[self.current_time_index].tolist(),
                self.input_matrix_2[self.current_time_index].tolist(),
                self.input_matrix_3[self.current_time_index].tolist()
            ]

            # Create the message
            msg = Float32MultiArray()
            # Flatten the nested list and assign it to msg.data
            msg.data = [item for sublist in combined_input for item in sublist]

            # Publish the message
            self.input_pub.publish(msg)

            # Log publication
            rospy.loginfo(
                "Published t={:.2f}, "
                "Max (Matrix 1): {:.2f}, "
                "Max (Matrix 2): {:.2f}, "
                "Max (Matrix 3): {:.2f}".format(
                    self.t[self.current_time_index],
                    self.input_matrix_1[self.current_time_index].max(),
                    self.input_matrix_2[self.current_time_index].max(),
                    self.input_matrix_3[self.current_time_index].max()
                )
            )

            self.current_time_index += 1
        else:
            # Stop timer when all slices are published
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
