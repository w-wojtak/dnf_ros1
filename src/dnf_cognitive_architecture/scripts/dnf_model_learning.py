#!/usr/bin/env python

import rospy
import threading
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from std_msgs.msg import Float32MultiArray
from datetime import datetime


class DNFModel:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('dnf_model_learning', anonymous=True)

        # Spatial and temporal parameters
        self.x_lim = 80
        self.t_lim = 15
        self.dx = 0.2
        self.dt = 0.1

        # Spatial and time grid
        self.x = np.arange(-self.x_lim, self.x_lim + self.dx, self.dx)
        self.t = np.arange(0, self.t_lim + self.dt, self.dt)

        self.u_sm_history = []
        self.u_sm_2_history = []
        self.u_d_history = []

        # Lock for thread safety
        self._lock = threading.Lock()

        # Subscriber
        self.subscriber = rospy.Subscriber(
            'input_matrices_combined', Float32MultiArray, self.input_callback, queue_size=10)

        # Time tracking
        self.time_counter = 0.0
        self.current_step = 1

        # Figure for plotting
        self.fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        self.ax1, self.ax2 = axes.flatten()
        self.line_u_sm_1, = self.ax1.plot(
            self.x, np.zeros_like(self.x), label="u_sm_1")
        self.line_u_sm_2, = self.ax2.plot(
            self.x, np.zeros_like(self.x), label="u_sm_2")

        for ax in [self.ax1, self.ax2]:
            ax.set_xlim(-self.x_lim, self.x_lim)
            ax.set_ylim(-2, 10)
            ax.set_xlabel("x")
            ax.set_ylabel("u(x)")
            ax.legend()

        self.ax1.set_title("Sequence Memory Field 1 (Robot)")
        self.ax2.set_title("Sequence Memory Field 2 (Human)")

        # Initialize fields
        self.h_0_sm = 0
        self.tau_h_sm = 20
        self.theta_sm = 1.5

        self.h_0_sm_2 = 0
        self.theta_sm_2 = 1.5

        self.kernel_pars_sm = (1, 0.7, 0.9)
        self.w_hat_sm = np.fft.fft(self.kernel_osc(*self.kernel_pars_sm))

        self.u_sm = self.h_0_sm * np.ones(np.shape(self.x))
        self.h_u_sm = self.h_0_sm * np.ones(np.shape(self.x))

        self.u_sm_2 = self.h_0_sm_2 * np.ones(np.shape(self.x))
        self.h_u_sm_2 = self.h_0_sm_2 * np.ones(np.shape(self.x))

        self.h_0_d = 0
        self.tau_h_d = 20
        self.theta_d = 1.5

        self.kernel_pars_d = (1, 0.7, 0.9)
        self.w_hat_d = np.fft.fft(self.kernel_osc(*self.kernel_pars_d))

        self.u_d = self.h_0_d * np.ones(np.shape(self.x))
        self.h_u_d = self.h_0_d * np.ones(np.shape(self.x))

    def input_callback(self, msg):
        received_data = np.array(msg.data)
        n = len(received_data) // 3

        input_agent1 = received_data[:n]
        input_agent2 = received_data[n:2 * n]

        rospy.loginfo(f"1ST INPUT MAX {max(input_agent1)}")
        rospy.loginfo(f"2ND INPUT MAX {max(input_agent2)}")

        self.time_counter += self.dt
        max_value = input_agent1.max()

        if 0.0 <= self.time_counter < 1.0:
            input_d = self.gaussian(0, 5.0, 2.0)
        else:
            input_d = 0.0

        f_d = np.heaviside(self.u_d - self.theta_d, 1)
        f_hat_d = np.fft.fft(f_d)
        conv_d = self.dx * \
            np.fft.ifftshift(np.real(np.fft.ifft(f_hat_d * self.w_hat_d)))

        f_sm = np.heaviside(self.u_sm - self.theta_sm, 1)
        f_hat_sm = np.fft.fft(f_sm)
        conv_sm = self.dx * \
            np.fft.ifftshift(np.real(np.fft.ifft(f_hat_sm * self.w_hat_sm)))

        f_sm_2 = np.heaviside(self.u_sm_2 - self.theta_sm_2, 1)
        f_hat_sm_2 = np.fft.fft(f_sm_2)
        conv_sm_2 = self.dx * \
            np.fft.ifftshift(np.real(np.fft.ifft(f_hat_sm_2 * self.w_hat_sm)))

        self.h_u_d += self.dt / self.tau_h_d * f_d
        self.u_d += self.dt * (-self.u_d + conv_d + input_d + self.h_u_d)

        self.h_u_sm += self.dt / self.tau_h_sm * f_sm
        self.u_sm += self.dt * (-self.u_sm + conv_sm +
                                input_agent1 + self.h_u_sm)

        self.h_u_sm_2 += self.dt / self.tau_h_sm * f_sm_2
        self.u_sm_2 += self.dt * \
            (-self.u_sm_2 + conv_sm_2 + input_agent2 + self.h_u_sm_2)

        input_positions = [-40, 0, 40]
        input_indices = [np.argmin(np.abs(self.x - pos))
                         for pos in input_positions]

        u_sm_values_at_positions = [self.u_sm[idx] for idx in input_indices]
        self.u_sm_history.append(u_sm_values_at_positions)

        u_sm_2_values_at_positions = [self.u_sm_2[idx]
                                      for idx in input_indices]
        self.u_sm_2_history.append(u_sm_2_values_at_positions)

        center_index = len(self.u_d) // 2
        self.u_d_history.append(self.u_d[center_index])

        if int(self.time_counter) >= self.current_step:
            rospy.loginfo(
                f"Time {self.current_step}: Input max = {max_value:.2f}")
            self.current_step += 1

        if self.time_counter >= self.t_lim:
            rospy.loginfo("Learning finished.")
            self.save_sequence_memory()
            rospy.signal_shutdown("Finished learning")

    def plt_func(self, _):
        with self._lock:
            self.line_u_sm_1.set_ydata(self.u_sm)
            self.line_u_sm_2.set_ydata(self.u_sm_2)
        return self.line_u_sm_1, self.line_u_sm_2

    def _plt(self):
        self.ani = anim.FuncAnimation(self.fig, self.plt_func, interval=100)
        plt.show()

    def kernel_osc(self, a, b, alpha):
        return a * (np.exp(-b * abs(self.x)) * ((b * np.sin(abs(alpha * self.x))) + np.cos(alpha * self.x)))

    def gaussian(self, center=0, amplitude=1.0, width=1.0):
        return amplitude * np.exp(-((self.x - center) ** 2) / (2 * width ** 2))

    def save_sequence_memory(self):
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        np.save(os.path.join(
            data_dir, f"u_sm_{timestamp}.npy"), self.u_sm_history)
        np.save(os.path.join(
            data_dir, f"u_sm_2_{timestamp}.npy"), self.u_sm_2_history)
        np.save(os.path.join(
            data_dir, f"u_d_{timestamp}.npy"), self.u_d_history)
        rospy.loginfo("Saved sequence memory to disk.")


if __name__ == "__main__":
    dnf_model = DNFModel()

    # Launch the plot in a separate thread
    plot_thread = threading.Thread(target=dnf_model._plt)
    plot_thread.start()

    rospy.spin()
