#!/usr/bin/env python
import matplotlib
matplotlib.use('Qt5Agg')  # Change to Qt5Agg backend
matplotlib.rcParams['figure.autolayout'] = True  # Add this line

import rospy
import threading
import numpy as np
import os
import matplotlib.pyplot as plt
plt.style.use('default')  # Use default style
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

        self.input_positions = [-60, -20, 20, 40]

        self.u_sm_history = []
        self.u_sm_2_history = []
        self.u_d_history = []

        # Lock for thread safety
        self._lock = threading.Lock()

        # Time tracking
        self.time_counter = 0.0
        self.current_step = 1

        # Initialize fields for sequence memory field 1
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

        # Initialize detection field BEFORE creating subscriber
        self.h_0_d = 0
        self.tau_h_d = 20
        self.theta_d = 1.5

        self.kernel_pars_d = (1, 0.7, 0.9)
        self.w_hat_d = np.fft.fft(self.kernel_osc(*self.kernel_pars_d))

        self.u_d = self.h_0_d * np.ones(np.shape(self.x))
        self.h_u_d = self.h_0_d * np.ones(np.shape(self.x))

        # Add these debug lines
        rospy.loginfo(f"Initial u_sm shape: {self.u_sm.shape}, values: min={np.min(self.u_sm)}, max={np.max(self.u_sm)}")
        rospy.loginfo(f"Initial u_sm_2 shape: {self.u_sm_2.shape}, values: min={np.min(self.u_sm_2)}, max={np.max(self.u_sm_2)}")

        # NOW create the subscriber after all fields are initialized
        self.subscriber = rospy.Subscriber(
            'input_matrices_combined', Float32MultiArray, self.input_callback, queue_size=10)

        # Setup the plot window
        self.fig = plt.figure(figsize=(10, 5))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        
        # Create line objects
        self.line1, = self.ax1.plot(self.x, self.u_sm, 'b-', label="u_sm")
        self.line2, = self.ax2.plot(self.x, self.u_d, 'r-', label="u_d")
        
        # Define object positions and labels
        object_positions = [-60, -20, 20, 40]
        object_labels = ['base', 'load', 'bearing', 'motor']

        object_all = [-60, -40, -20, 0, 20, 40, 60]
        object_labels_all = ['base', 'blue box', 'load', 'tool 1', 'bearing', 'motor', 'tool 2']
        
        # Set up the axes
        for ax in [self.ax1]:
            ax.set_xlim(-self.x_lim, self.x_lim)
            ax.set_ylim(-2, 6)
            ax.set_xlabel("Objects")
            ax.set_ylabel("u(x)")
            ax.grid(True)
            # ax.legend()
            
            # Set custom x-ticks at object positions
            ax.set_xticks(object_all)
            ax.set_xticklabels(object_labels_all)
            # Rotate labels for better readability
            ax.tick_params(axis='x', rotation=45)
            
            # Add vertical lines at object positions (optional)
            for pos in object_all:
                ax.axvline(x=pos, color='gray', linestyle='--', alpha=0.3)
        
        self.ax1.set_title("Sequence Memory Field")
        self.ax2.set_title("Task Duration Field")

        self.ax2.set_xticks([0])
        self.ax2.set_xticklabels(['task start'])
        self.ax2.tick_params(axis='x', rotation=0) 
        self.ax2.set_ylim(-2, 6)
        self.ax2.grid(True)


        # Create a timer for updating the plot
        self.timer = self.fig.canvas.new_timer(interval=100)  # 100ms interval
        self.timer.add_callback(self.update_plot)
        self.timer.start()

        rospy.loginfo("DNF Model initialized successfully")

    def input_callback(self, msg):
        try:
            with self._lock:  # Added proper thread safety
                received_data = np.array(msg.data)
                n = len(received_data) // 3

                input_agent1 = received_data[:n]
                input_agent2 = received_data[n:2 * n]

                rospy.logdebug(f"1ST INPUT MAX {max(input_agent1)}")
                rospy.logdebug(f"2ND INPUT MAX {max(input_agent2)}")

                self.time_counter += self.dt
                max_value = input_agent1.max()

                # Input for detection field
                if 0.0 <= self.time_counter < 1.0:
                    input_d = self.gaussian(0, 5.0, 2.0)
                else:
                    # Fixed: should be array, not scalar
                    input_d = np.zeros_like(self.x)

                # Detection field dynamics
                f_d = np.heaviside(self.u_d - self.theta_d, 1)
                f_hat_d = np.fft.fft(f_d)
                conv_d = self.dx * \
                    np.fft.ifftshift(
                        np.real(np.fft.ifft(f_hat_d * self.w_hat_d)))

                # Sequence memory field 1 dynamics
                f_sm = np.heaviside(self.u_sm - self.theta_sm, 1)
                f_hat_sm = np.fft.fft(f_sm)
                conv_sm = self.dx * \
                    np.fft.ifftshift(
                        np.real(np.fft.ifft(f_hat_sm * self.w_hat_sm)))

                # Sequence memory field 2 dynamics
                f_sm_2 = np.heaviside(self.u_sm_2 - self.theta_sm_2, 1)
                f_hat_sm_2 = np.fft.fft(f_sm_2)
                conv_sm_2 = self.dx * \
                    np.fft.ifftshift(
                        np.real(np.fft.ifft(f_hat_sm_2 * self.w_hat_sm)))

                # Update fields
                self.h_u_d += self.dt / self.tau_h_d * f_d
                self.u_d += self.dt * \
                    (-self.u_d + conv_d + input_d + self.h_u_d)

                self.h_u_sm += self.dt / self.tau_h_sm * f_sm
                self.u_sm += self.dt * (-self.u_sm + conv_sm +
                                        input_agent1 + self.h_u_sm)

                self.h_u_sm_2 += self.dt / self.tau_h_sm * f_sm_2
                self.u_sm_2 += self.dt * \
                    (-self.u_sm_2 + conv_sm_2 + input_agent2 + self.h_u_sm_2)
                
                rospy.loginfo(f"Updated values - u_sm max={np.max(self.u_sm):.2f}, u_sm_2 max={np.max(self.u_sm_2):.2f}")

                # Store history at specific positions
                # input_positions = [-60, -20, 20, 40]
                input_indices = [np.argmin(np.abs(self.x - pos))
                                 for pos in self.input_positions]

                u_sm_values_at_positions = [self.u_sm[idx]
                                            for idx in input_indices]
                self.u_sm_history.append(u_sm_values_at_positions)

                u_sm_2_values_at_positions = [self.u_sm_2[idx]
                                              for idx in input_indices]
                self.u_sm_2_history.append(u_sm_2_values_at_positions)

                center_index = len(self.u_d) // 2
                self.u_d_history.append(self.u_d[center_index])

                # Progress logging
                if int(self.time_counter) >= self.current_step:
                    rospy.loginfo(
                        f"Time {self.current_step}: Input max = {max_value:.2f}")
                    self.current_step += 1

                # Check if learning is finished
                if self.time_counter >= self.t_lim:
                    rospy.loginfo("Learning finished.")
                    self.save_sequence_memory()
                    rospy.signal_shutdown("Finished learning")
                    

        except Exception as e:
            rospy.logerr(f"Error in input_callback: {str(e)}")

    def update_plot(self):
        """Update plot data without blocking"""
        try:
            with self._lock:
                # Update line data
                self.line1.set_ydata(self.u_sm)
                self.line2.set_ydata(self.u_sm_2)
                
                # Update display
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
        except Exception as e:
            rospy.logerr(f"Plot update error: {str(e)}")

    def kernel_osc(self, a, b, alpha):
        """Oscillatory kernel function"""
        return a * (np.exp(-b * np.abs(self.x)) *
                    (b * np.sin(np.abs(alpha * self.x)) + np.cos(alpha * self.x)))

    def gaussian(self, center=0, amplitude=1.0, width=1.0):
        """Gaussian input function"""
        return amplitude * np.exp(-((self.x - center) ** 2) / (2 * width ** 2))

    def save_sequence_memory(self):
        """Save sequence memory data to files"""
        try:
            # Get the workspace root directory (two levels up from the scripts directory)
            current_dir = os.path.dirname(os.path.abspath(__file__))  # scripts directory
            workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))  # up to workspace root
            data_dir = os.path.join(workspace_root, "data")
            
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                rospy.loginfo(f"Created data directory at {data_dir}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Add debug prints
            rospy.loginfo(f"Saving to directory: {data_dir}")
            rospy.loginfo(f"Current u_sm shape: {self.u_sm.shape}")
            
            # Save with full paths and error checking
            u_sm_path = os.path.join(data_dir, f"u_sm_{timestamp}.npy")
            u_sm_2_path = os.path.join(data_dir, f"u_sm_2_{timestamp}.npy")
            u_d_path = os.path.join(data_dir, f"u_d_{timestamp}.npy")
            
            np.save(u_sm_path, self.u_sm)
            np.save(u_sm_2_path, self.u_sm_2)
            np.save(u_d_path, self.u_d)
            
            # Verify files were created
            if os.path.exists(u_sm_path):
                rospy.loginfo(f"Successfully saved u_sm to {u_sm_path}")
            else:
                rospy.logerr(f"Failed to save u_sm to {u_sm_path}")
                
            if os.path.exists(u_sm_2_path):
                rospy.loginfo(f"Successfully saved u_sm_2 to {u_sm_2_path}")
            else:
                rospy.logerr(f"Failed to save u_sm_2 to {u_sm_2_path}")
                
            if os.path.exists(u_d_path):
                rospy.loginfo(f"Successfully saved u_d to {u_d_path}")
            else:
                rospy.logerr(f"Failed to save u_d to {u_d_path}")

        except Exception as e:
            rospy.logerr(f"Error saving sequence memory: {str(e)}")
            import traceback
            rospy.logerr(traceback.format_exc())

            
    def plot_activity_evolution(self, save_plot: bool = True, show_plot: bool = True):
        try:

            u_sm_hist = np.array(self.u_sm_history)
            u_sm_2_hist = np.array(self.u_sm_2_history)
            u_d_hist = np.array(self.u_d_history)
            time_steps = np.arange(len(u_sm_hist)) * self.dt

            fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

            # Define object names for each position
            object_names = {
                -60: 'base',
                -20: 'load',
                20: 'bearing',
                40: 'motor'
            }

            # Plot u_sm (Agent 1)
            for i, pos in enumerate(self.input_positions):
                axes[0].plot(time_steps, u_sm_hist[:, i], label=object_names[pos])
            axes[0].set_title('u_sm (Agent 1) over time at input positions')
            axes[0].set_ylabel('u_sm')
            axes[0].legend()
            axes[0].grid(True)

            # Plot u_sm_2 (Agent 2)
            for i, pos in enumerate(self.input_positions):
                axes[1].plot(time_steps, u_sm_2_hist[:, i], label=object_names[pos])
            axes[1].set_title('u_sm_2 (Agent 2) over time at input positions')
            axes[1].set_ylabel('u_sm_2')
            axes[1].legend()
            axes[1].grid(True)


            # Plot u_d (center)
            axes[2].plot(time_steps, u_d_hist, label='center x=0', color='black')
            axes[2].set_title('u_d over time at center position')
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('u_d')
            axes[2].legend()
            axes[2].grid(True)

            plt.tight_layout()

            # Save to disk
            if save_plot:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
                data_dir = os.path.join(workspace_root, "data")
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_path = os.path.join(data_dir, f"evolution_plot_{timestamp}.png")
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




    def shutdown_hook(self):
        """Clean shutdown"""
        rospy.loginfo("Shutting down DNF Model...")
        # Don't close plots here
        pass


if __name__ == "__main__":
    try:
        dnf_model = DNFModel()
        
        def shutdown_handler():
            rospy.loginfo("Shutdown handler called")
            dnf_model.save_sequence_memory()
            plt.close('all')
        
        rospy.on_shutdown(shutdown_handler)
        rospy.loginfo("DNF Model started. Waiting for input...")
        
        # Create a separate thread for ROS spinning
        import threading
        ros_thread = threading.Thread(target=rospy.spin)
        ros_thread.daemon = True
        ros_thread.start()
        
        # Show plot in main thread
        plt.show()
            
    except rospy.ROSInterruptException:
        rospy.loginfo("DNF Model interrupted by user")
    except Exception as e:
        rospy.logerr(f"Fatal error in DNF Model: {str(e)}")
    finally:
        rospy.loginfo("Entering finally block")
        if 'dnf_model' in locals():
            rospy.loginfo("Saving data in finally block")
            dnf_model.save_sequence_memory()
            dnf_model.plot_activity_evolution(save_plot=False, show_plot=True)
        plt.close('all')

