#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32MultiArray, String
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import threading
from datetime import datetime
import os
import time
from scipy.ndimage import gaussian_filter1d  # type: ignore


def find_bump_centers(u_array, theta, bump_half_width=25):
    centers = []
    u_copy = u_array.copy()

    while np.max(u_copy) > theta:
        max_idx = np.argmax(u_copy)
        centers.append(max_idx)

        start = max(0, max_idx - bump_half_width)
        end = min(len(u_copy), max_idx + bump_half_width + 1)
        u_copy[start:end] = 0.0

    return centers


class DNFModelWM:
    def __init__(self):
        rospy.init_node("dnf_model_recall", anonymous=True)

        # Get the 'trial_number' parameter (default value is 1)
        self.trial_number = rospy.get_param('~trial_number', 1)

        # Log the trial number
        rospy.loginfo(
            f"Recall node started with trial_number: {self.trial_number}")

        # Persistently track threshold crossings
        # self.threshold_crossed = {pos: False for pos in [-40, 0, 40]}

        self.POSITION_TO_OBJECT = {
            -60: 'base',
            -20: 'load',
            20: 'bearing',
            40: 'motor'
        }

        self.threshold_crossed = {pos: False for pos in self.POSITION_TO_OBJECT.keys()}

        # Spatial and temporal parameters
        self.x_lim = 80
        self.t_lim = 15
        self.dx = 0.2
        self.dt = 0.1

        # Spatial grid
        self.x = np.arange(-self.x_lim, self.x_lim + self.dx, self.dx)
        rospy.loginfo(f"Spatial grid size: {len(self.x)}")

        # Lock for threading
        self._lock = threading.Lock()

        # Publisher
        self.publisher = rospy.Publisher(
            'threshold_crossings', Float32MultiArray, queue_size=10)
        
        self.gripper_pub = rospy.Publisher('/gripper/command', String, queue_size=10)

        self.gripper_status_sub = rospy.Subscriber('/gripper/status', String, self.gripper_status_callback)

        # Variable to store the latest input slice
        self.latest_input_slice = np.zeros_like(self.x)

        # Initialize figure and axes for plotting
        plt.ion()  # Enable interactive mode
        self.fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6 = axes.flatten()

        # Define object positions and labels
        object_all = [-60, -40, -20, 0, 20, 40, 60]
        object_labels_all = ['base', 'blue box', 'load', 'tool 1', 'bearing', 'motor', 'tool 2']

        # Plot for u_act on ax1
        self.line_act, = self.ax1.plot(
            self.x, np.zeros_like(self.x), label="u_act")
        self.ax1.set_ylabel("u_act(x)")
        self.ax1.set_title("Action Onset Field")
        self.ax1.legend()

        # Plot for u_wm on ax3
        self.line_wm, = self.ax3.plot(
            self.x, np.zeros_like(self.x), label="u_wm")
        self.ax3.set_ylabel("u_wm(x)")
        self.ax3.set_title("Working Memory Field")
        self.ax3.legend()

        # Plot for u_f1 on ax4
        self.line_f1, = self.ax4.plot(
            self.x, np.zeros_like(self.x), label="u_f1")
        self.ax4.set_ylabel("u_f1(x)")
        self.ax4.set_title("Feedback 1 Field")
        self.ax4.legend()

        # Plot for u_f2 on ax5
        self.line_f2, = self.ax5.plot(
            self.x, np.zeros_like(self.x), label="u_f2")
        self.ax5.set_ylabel("u_f2(x)")
        self.ax5.set_title("Feedback 2 Field")
        self.ax5.legend()

        # Plot for u_sim on ax2
        self.line_sim, = self.ax2.plot(
            self.x, np.zeros_like(self.x), label="u_sim")
        self.ax2.set_ylabel("u_sim(x)")
        self.ax2.set_title("Simulation Field")
        self.ax2.legend()

        # Plot for u_error on ax6
        self.line_error, = self.ax6.plot(
            self.x, np.zeros_like(self.x), label="u_error")
        self.ax6.set_ylabel("u_error(x)")
        self.ax6.set_title("Error Field")
        self.ax6.legend()

        # Apply the same formatting to all axes
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
            ax.set_xlim(-self.x_lim, self.x_lim)
            ax.set_ylim(-5, 5)  # Adjust based on expected amplitude
            ax.set_xlabel("Objects")
            ax.grid(True)
            
            # Set custom x-ticks at object positions
            ax.set_xticks(object_all)
            ax.set_xticklabels(object_labels_all)
            # Rotate labels for better readability
            ax.tick_params(axis='x', rotation=45)
            
            # Add vertical lines at object positions (optional)
            for pos in object_all:
                ax.axvline(x=pos, color='gray', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.show(block=False)

        # Initialize fields with default values first
        self.h_d_initial = 0
        self.u_act = np.zeros_like(self.x)
        self.h_u_act = np.zeros_like(self.x)
        self.input_action_onset = np.zeros_like(self.x)
        
        self.u_sim = np.zeros_like(self.x)
        self.h_u_sim = np.zeros_like(self.x)
        self.input_action_onset_2 = np.zeros_like(self.x)

        # Try to load data
        try:
            self.u_d = load_task_duration().flatten()
            self.h_d_initial = max(self.u_d)
            rospy.loginfo(f"Loaded u_d size: {len(self.u_d)}")

            if self.trial_number == 1:
                # Ensure it's 1D and shift as needed
                # It should be + 1.5 (=theta), but I used + 1.54 because recall was a bit delayed
                self.u_act = load_sequence_memory().flatten() - self.h_d_initial + 1.54
                self.input_action_onset = load_sequence_memory().flatten()
                self.h_u_act = -self.h_d_initial * np.ones(np.shape(self.x)) + 1.54

                self.u_sim = load_sequence_memory().flatten() - self.h_d_initial + 1.54
                self.input_action_onset_2 = load_sequence_memory().flatten()
                self.h_u_sim = -self.h_d_initial * np.ones(np.shape(self.x)) + 1.54
            else:
                data_dir = os.path.join(os.getcwd(), 'dnf_architecture_extended/data_basic')
                rospy.loginfo(f"Loading from {data_dir}")
                latest_h_amem_file = get_latest_file(data_dir, 'h_amem')
                if latest_h_amem_file:
                    latest_h_amem = np.load(latest_h_amem_file, allow_pickle=True)
                    self.u_act = load_sequence_memory().flatten() - self.h_d_initial + 1.54 + latest_h_amem
                    self.input_action_onset = load_sequence_memory().flatten() + latest_h_amem
                    self.h_u_act = -self.h_d_initial * np.ones(np.shape(self.x)) + 1.54

                    self.u_sim = load_sequence_memory().flatten() - self.h_d_initial + 1.54
                    self.input_action_onset_2 = load_sequence_memory().flatten()
                    self.h_u_sim = -self.h_d_initial * np.ones(np.shape(self.x)) + 1.54

        except IOError as e:
            rospy.loginfo(f"No previous sequence memory found: {e}")
            # Fields are already initialized with zeros above

        # Parameters specific to working memory
        self.h_0_wm = -1.0
        self.theta_wm = 0.8

        self.kernel_pars_wm = (1.75, 0.5, 0.8)
        self.w_hat_wm = np.fft.fft(self.kernel_osc(*self.kernel_pars_wm))

        # initialization
        self.u_wm = self.h_0_wm * np.ones(np.shape(self.x))
        self.h_u_wm = self.h_0_wm * np.ones(np.shape(self.x))

        # Parameters specific to action onset
        self.tau_h_act = 20
        self.theta_act = 1.5

        self.tau_h_sim = 10
        self.theta_sim = 1.5

        self.theta_error = 1.5

        self.kernel_pars_act = (1.5, 0.8, 0.0)
        self.w_hat_act = np.fft.fft(self.kernel_gauss(*self.kernel_pars_act))

        self.kernel_pars_sim = (1.7, 0.8, 0.7)
        self.w_hat_sim = np.fft.fft(self.kernel_gauss(*self.kernel_pars_sim))

        # feedback fields - decision fields, similar to u_act
        self.h_f = -1.0
        self.w_hat_f = self.w_hat_act

        self.tau_h_f = self.tau_h_act
        self.theta_f = self.theta_act

        self.u_f1 = self.h_f * np.ones(np.shape(self.x))
        self.u_f2 = self.h_f * np.ones(np.shape(self.x))

        self.u_error = self.h_f * np.ones(np.shape(self.x))

        self.u_act_history = []  # Lists to store values at each time step
        self.u_sim_history = []
        self.u_wm_history = []
        self.u_f1_history = []
        self.u_f2_history = []
        self.u_error_history = []

        # Variable to store the latest input slices - initialize all three
        self.input_agent1 = np.zeros_like(self.x)
        self.input_agent2 = np.zeros_like(self.x)
        self.input_agent_robot_feedback = np.zeros_like(self.x)
        

        # BUMP CENTERS IN U_SIM
        self.sim_centers = find_bump_centers(self.u_sim, 1.0)
        self.x_centers = [round(self.x[idx]) for idx in self.sim_centers]

        # self.get_logger().info(f"BUMP CENTERS: {self.x_centers}")
        rospy.loginfo(f"BUMP CENTERS: {self.x_centers}")

        # initialize h level for the adaptation
        self.h_u_amem = np.zeros(np.shape(self.x))
        self.beta_adapt = 0.01

        # Create subscriber and timer AFTER all fields are initialized
        self.subscription = rospy.Subscriber(
            'input_matrices_combined',
            Float32MultiArray,
            self.process_inputs,
            queue_size=10
        )

        self.speech_sub = rospy.Subscriber(
            '/mock_speech_recognition/command',
            String,
            self.speech_command_callback,
            queue_size=10
        )

        # Timer to publish every 1 second
        self.timer = rospy.Timer(rospy.Duration(1.0), self.timer_callback)

        self.speech_input_next_counter = 0

        self.input_robot_feedback = 0
        self.robot_feedback_counter = 0  # counts time steps remaining for feedback
        self.robot_feedback_center = None  # position to center the Gaussian

        self.pending_robot_feedback = []

    def gripper_status_callback(self, msg):
        object_name = msg.data
        for pos, name in self.POSITION_TO_OBJECT.items():
            if name == object_name:
                with self._lock:
                    self.robot_feedback_center = pos
                    self.robot_feedback_counter = 10  # Fixed number of steps
                    rospy.loginfo(f"Set robot feedback: center={pos}, counter={self.robot_feedback_counter}")
                break
        else:
            rospy.logwarn(f"Unknown object name from gripper: {object_name}")

    def timer_callback(self, event):
        """Timer callback to process inputs periodically."""
        try:
            self.perform_recall()
        except Exception as e:
            rospy.logerr(f"Error in timer_callback: {e}")

    def process_inputs(self, msg):
        """Process recall by receiving msg from subscription."""
        try:
            # Handle incoming message from subscriber
            received_data = np.array(msg.data)
            
            # Get the size of each input (assuming 3 equal parts)
            n = len(received_data) // 3

            # rospy.loginfo(f"Debug sizes:")
            # rospy.loginfo(f"self.x size: {len(self.x)}")
            # rospy.loginfo(f"Received data size: {len(received_data)}")
            # rospy.loginfo(f"Each input size (n): {n}")
            
            with self._lock:
                # If input size doesn't match field size, interpolate
                if n != len(self.x):
                    # Create interpolation indices
                    x_input = np.linspace(-self.x_lim, self.x_lim, n)
                    
                    # Interpolate each input to match field size
                    self.input_agent1 = np.interp(self.x, x_input, received_data[:n])
                    self.input_agent2 = np.interp(self.x, x_input, received_data[n:2*n])
                    self.input_agent_robot_feedback = np.interp(self.x, x_input, received_data[2*n:])
                else:
                    # Direct assignment if sizes match
                    self.input_agent1 = received_data[:n]
                    self.input_agent2 = received_data[n:2*n]
                    self.input_agent_robot_feedback = received_data[2*n:]
            
            # Handle the logic for both subscription and timer
            # self.perform_recall()
        except Exception as e:
            rospy.logerr(f"Error in process_inputs: {e}")

    def speech_command_callback(self, msg):
        command = msg.data
        rospy.loginfo("Received speech command: '%s'" % command)
        if command == "start":
            rospy.loginfo("Starting trial!")
            # Example: set a flag or counter for "start" if needed
            # with self._lock:
            #     self.speech_input_start_counter = 10
            # self.start_trial()  # (implement this if needed)
        elif command == "finished":
            rospy.loginfo("Trial finished!")
            # Example: set a flag or counter for "finished" if needed
            # with self._lock:
            #     self.speech_input_finished_counter = 10
            # self.finish_trial()  # (implement this if needed)
        elif command == "next":
            rospy.loginfo("Next object requested!")
            with self._lock:
                self.speech_input_next_counter = 10  # Set for 10 time steps
            # self.next_object()  # (implement this if needed)
        else:
            rospy.logwarn("Unknown speech command: '%s'" % command)

    def perform_recall(self):
        with self._lock:  # Thread safety

                            
            self.input_amplitude = 5.0
            self.input_width = 2.0

            if self.speech_input_next_counter > 0:
                # Extract values of u_sim at bump centers (indices)
                values_at_centers = [self.u_sim[idx] for idx in self.sim_centers]

                # Find the index of the max value among those centers
                max_idx = values_at_centers.index(max(values_at_centers))

                # Get the center location with highest value
                max_center = round(self.x[self.sim_centers[max_idx]])

                # Log or use the result
                rospy.loginfo(
                    f"Highest bump center: {max_center} with value {values_at_centers[max_idx]:.3f}")

                # Create Gaussian at highest center
                gauss_at_peak = self.input_amplitude * \
                    np.exp(-((self.x - max_center) ** 2) /
                        (2 * self.input_width**2))

                input_human_feedback = gauss_at_peak

                # Decrement the counter
                self.speech_input_next_counter -= 1

            else:
                input_human_feedback = 0

            # Robot feedback logic
            if self.robot_feedback_counter > 0 and self.robot_feedback_center is not None:
                rospy.loginfo(f"APPLY ROBOT INPUT, counter: {self.robot_feedback_counter}")
                gauss_at_robot = self.input_amplitude * np.exp(
                    -((self.x - self.robot_feedback_center) ** 2) / (2 * self.input_width ** 2)
                )
                self.input_robot_feedback = gauss_at_robot
                self.robot_feedback_counter -= 1
            else:
                self.input_robot_feedback = 0
                
        
            # Use the stored inputs directly (no need to split again)
            f_f1 = np.heaviside(self.u_f1 - self.theta_f, 1)
            f_hat_f1 = np.fft.fft(f_f1)
            conv_f1 = self.dx * \
                np.fft.ifftshift(np.real(np.fft.ifft(f_hat_f1 * self.w_hat_f)))

            f_f2 = np.heaviside(self.u_f2 - self.theta_f, 1)
            f_hat_f2 = np.fft.fft(f_f2)
            conv_f2 = self.dx * \
                np.fft.ifftshift(np.real(np.fft.ifft(f_hat_f2 * self.w_hat_f)))

            f_act = np.heaviside(self.u_act - self.theta_act, 1)
            f_hat_act = np.fft.fft(f_act)
            conv_act = self.dx * \
                np.fft.ifftshift(np.real(np.fft.ifft(f_hat_act * self.w_hat_act)))

            f_sim = np.heaviside(self.u_sim - self.theta_sim, 1)
            f_hat_sim = np.fft.fft(f_sim)
            conv_sim = self.dx * \
                np.fft.ifftshift(np.real(np.fft.ifft(f_hat_sim * self.w_hat_sim)))

            f_wm = np.heaviside(self.u_wm - self.theta_wm, 1)
            f_hat_wm = np.fft.fft(f_wm)
            conv_wm = self.dx * \
                np.fft.ifftshift(np.real(np.fft.ifft(f_hat_wm * self.w_hat_wm)))

            f_error = np.heaviside(self.u_error - self.theta_error, 1)
            f_hat_error = np.fft.fft(f_error)
            conv_error = self.dx * \
                np.fft.ifftshift(
                    np.real(np.fft.ifft(f_hat_error * self.w_hat_act)))

            # Update field states
            self.h_u_act += self.dt / self.tau_h_act
            self.h_u_sim += self.dt / self.tau_h_sim

            self.u_act += self.dt * (-self.u_act + conv_act + self.input_action_onset +
                                    self.h_u_act - 6.0 * f_wm * conv_wm)

            self.u_sim += self.dt * (-self.u_sim + conv_sim + self.input_action_onset_2 +
                                    self.h_u_sim - 6.0 * f_wm * conv_wm)

            self.u_wm += self.dt * \
                (-self.u_wm + conv_wm + 6*((f_f1*self.u_f1)*(f_f2*self.u_f2)) + self.h_u_wm)

            # Use the stored inputs
            # self.u_f1 += self.dt * (-self.u_f1 + conv_f1 + self.input_agent_robot_feedback +
            #                         self.h_f - 1 * f_wm * conv_wm)

            # self.u_f2 += self.dt * (-self.u_f2 + conv_f2 + self.input_agent2 +
            #                         self.h_f - 1 * f_wm * conv_wm)

            self.u_f1 += self.dt * (-self.u_f1 + conv_f1 + self.input_robot_feedback  +
                        self.h_f - 1 * f_wm * conv_wm)
            
            self.u_f2 += self.dt * (-self.u_f2 + conv_f2 + input_human_feedback +
                        self.h_f - 1 * f_wm * conv_wm)

            self.u_error += self.dt * (-self.u_error + conv_error +
                                    self.h_f - 2 * f_sim * conv_sim)

            self.h_u_amem += self.beta_adapt*(1 - (f_f2 * f_f1)) * (f_f1 - f_f2)

            # Rest of the method remains the same...
            # List of input positions where we previously applied inputs
            input_positions = list(self.POSITION_TO_OBJECT.keys())

            # Convert `input_positions` to indices in `self.x`
            input_indices = [np.argmin(np.abs(self.x - pos))
                            for pos in input_positions]

            # Store the values at the specified positions
            u_act_values_at_positions = [self.u_act[idx] for idx in input_indices]
            self.u_act_history.append(u_act_values_at_positions)

            u_sim_values_at_positions = [self.u_sim[idx] for idx in input_indices]
            self.u_sim_history.append(u_sim_values_at_positions)

            u_wm_values_at_positions = [self.u_wm[idx] for idx in input_indices]
            self.u_wm_history.append(u_wm_values_at_positions)

            u_f1_values_at_positions = [self.u_f1[idx] for idx in input_indices]
            self.u_f1_history.append(u_f1_values_at_positions)

            u_f2_values_at_positions = [self.u_f2[idx] for idx in input_indices]
            self.u_f2_history.append(u_f2_values_at_positions)

            # Check `u_act` values at exact input indices for threshold crossings
            for i, idx in enumerate(input_indices):
                position = input_positions[i]

                # Only proceed if the threshold has not yet been crossed for this input position
                if not self.threshold_crossed[position] and self.u_act[idx] > self.theta_act:
                    # Map position to object name
                    object_name = self.POSITION_TO_OBJECT.get(position)
                    if object_name:
                        gripper_msg = String()
                        gripper_msg.data = object_name
                        self.gripper_pub.publish(gripper_msg)
                        rospy.loginfo(f"Published gripper command: {object_name}")
                    else:
                        rospy.logwarn(f"No object mapped for position {position}")

                    self.threshold_crossed[position] = True
                    # # Debugging line
                    # print(
                    #     f"Threshold crossed at position {position} with u_act = {self.u_act[idx]}")
                    # threshold_msg = Float32MultiArray()
                    # threshold_msg.data = [float(position)]
                    # self.publisher.publish(threshold_msg)
                    # self.threshold_crossed[position] = True


    def update_plot(self):
        """Update plot data without blocking"""
        try:
            with self._lock:
                # Update the plot with the latest data for both fields
                self.line_act.set_ydata(self.u_act)
                self.line_sim.set_ydata(self.u_sim)
                self.line_wm.set_ydata(self.u_wm)
                self.line_f1.set_ydata(self.u_f1)
                self.line_f2.set_ydata(self.u_f2)
                self.line_error.set_ydata(self.u_error)
                
                # Update timestep in the title
                timestep = len(self.u_act_history)
                self.fig.suptitle(f"Time: {timestep}", fontsize=16)
                self.fig.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
                
                # Update display
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
        except Exception as e:
            rospy.logerr(f"Plot update error: {e}")

    def kernel_osc(self, a, b, alpha):
        return a * (np.exp(-b * abs(self.x)) * ((b * np.sin(abs(alpha * self.x))) + np.cos(alpha * self.x)))

    def kernel_gauss(self, a_ex, s_ex, w_in):
        return a_ex * np.exp(-0.5 * self.x ** 2 / s_ex ** 2) - w_in

    def save_working_memory(self):
        # Create directory if it doesn't exist
        data_dir = "/home/robotica/dnf_ros1/data_basic"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Get current date and time for file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{data_dir}/working_memory_{timestamp}.npy"

        # Save u_wm data as a .npy file
        np.save(filename, self.u_wm)
        print(f"Working memory saved to {filename}")

    def save_history(self):
        data_dir = "/home/robotica/dnf_ros1/data_basic"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        rospy.loginfo(f"SAVING HISTORY to {data_dir}")
        rospy.loginfo(f"SAVING HISTORY SIZE U ACT {len(self.u_act_history)}")

        # Get current date and time for file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save data as a .npy files
        filename_act = f"{data_dir}/act_history_{timestamp}.npy"
        np.save(filename_act, self.u_act_history)

        filename_sim = f"{data_dir}/sim_history_{timestamp}.npy"
        np.save(filename_sim, self.u_sim_history)

        filename_wm = f"{data_dir}/wm_history_{timestamp}.npy"
        np.save(filename_wm, self.u_wm_history)

        filename_f1 = f"{data_dir}/f1_history_{timestamp}.npy"
        np.save(filename_f1, self.u_f1_history)

        filename_f2 = f"{data_dir}/f2_history_{timestamp}.npy"
        np.save(filename_f2, self.u_f2_history)

        self.h_u_amem = gaussian_filter1d(self.h_u_amem, sigma=5)

        filename_h_amem = f"{data_dir}/h_amem_{timestamp}.npy"
        np.save(filename_h_amem, self.h_u_amem)

        rospy.loginfo("HISTORY SAVED.")


    def plot_activity_evolution(self, save_plot: bool = True, show_plot: bool = True):
        try:
            # Convert histories to numpy arrays
            u_act_hist = np.array(self.u_act_history)
            u_sim_hist = np.array(self.u_sim_history)
            u_wm_hist = np.array(self.u_wm_history)
            u_f1_hist = np.array(self.u_f1_history)
            u_f2_hist = np.array(self.u_f2_history)
            time_steps = np.arange(len(u_act_hist)) * self.dt

            # Define object names for each position
            object_names = {
                -60: 'base',
                -20: 'load',
                20: 'bearing',
                40: 'motor'
            }
            input_positions = list(object_names.keys())

            fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)

            # Plot each field
            for i, pos in enumerate(input_positions):
                axes[0].plot(time_steps, u_act_hist[:, i], label=object_names[pos])
                axes[1].plot(time_steps, u_sim_hist[:, i], label=object_names[pos])
                axes[2].plot(time_steps, u_wm_hist[:, i], label=object_names[pos])
                axes[3].plot(time_steps, u_f1_hist[:, i], label=object_names[pos])
                axes[4].plot(time_steps, u_f2_hist[:, i], label=object_names[pos])

            axes[0].set_title('u_act over time at input positions')
            axes[1].set_title('u_sim over time at input positions')
            axes[2].set_title('u_wm over time at input positions')
            axes[3].set_title('u_f1 over time at input positions')
            axes[4].set_title('u_f2 over time at input positions')

            for ax in axes:
                ax.set_ylabel('Activity')
                ax.legend()
                ax.grid(True)
            axes[-1].set_xlabel('Time (s)')

            plt.tight_layout()

            # Save to disk
            if save_plot:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
                data_dir = os.path.join(workspace_root, "data_basic")
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_path = os.path.join(data_dir, f"activity_evolution_{timestamp}.png")
                fig.savefig(plot_path)
                rospy.loginfo(f"Saved activity evolution plot to {plot_path}")

            # Show the plot
            if show_plot:
                plt.show()
            else:
                plt.close(fig)

        except Exception as e:
            rospy.logerr(f"Error plotting activity evolution: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())


def load_sequence_memory(filename=None):
    data_dir = "/home/robotica/dnf_ros1/data_basic"
    if filename is None:
        # Filter files with the "sequence_memory_" prefix
        files = [f for f in os.listdir(data_dir) if f.startswith(
            "u_sm_") and f.endswith('.npy')]

        if not files:
            raise IOError(
                "No 'u_sm_' files found in the 'data' folder.")

        # Get the latest file by modification time
        latest_file = max([os.path.join(data_dir, f)
                          for f in files], key=os.path.getmtime)
        filename = latest_file

    data = np.load(filename)
    print(f"Loaded sequence memory from {filename}")

    # Ensure data is 1D
    data = data.flatten()

    return data



def load_task_duration(filename=None):
    data_dir = "/home/robotica/dnf_ros1/data_basic"
    if filename is None:
        # Filter files with the "sequence_memory_" prefix
        # files = [f for f in os.listdir(data_dir) if f.startswith(
        #     "task_duration_") and f.endswith('.npy')]
        files = [f for f in os.listdir(data_dir) if f.startswith(
            "u_d_") and f.endswith('.npy')]

        if not files:
            raise IOError(
                "No 'task_duration_' files found in the 'data' folder.")

        # Get the latest file by modification time
        latest_file = max([os.path.join(data_dir, f)
                          for f in files], key=os.path.getmtime)
        filename = latest_file

    # Load the data from the selected file
    data = np.load(filename)
    print(f"Loaded sequence memory from {filename}")

    # Ensure data is 1D
    data = data.flatten()

    # Print size and max value of the loaded data
    print(f"Data size: {data.size}")
    print(f"Max value: {data.max()}")

    return data


def get_latest_file(data_dir, pattern):
    """Retrieve the latest file in the data directory matching the pattern."""
    files = [f for f in os.listdir(data_dir) if f.startswith(
        pattern) and f.endswith('.npy')]
    if not files:
        return None
    # Sort files by modified time
    files.sort(key=lambda f: os.path.getmtime(
        os.path.join(data_dir, f)), reverse=True)
    return os.path.join(data_dir, files[0])


def main():
    try:
        node = DNFModelWM()
        rospy.loginfo("DNF Model WM started. Waiting for input...")

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            node.update_plot()
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.save_history()
            node.plot_activity_evolution(save_plot=True, show_plot=True)  
        # plt.close('all')  # <-- REMOVE THIS LINE

    # Show the plot after ROS is fully shut down
    import matplotlib.pyplot as plt
    plt.show()

if __name__ == '__main__':
    main()
