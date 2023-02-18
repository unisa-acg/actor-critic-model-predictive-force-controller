import torch
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped, PointStamped
from std_msgs.msg import String
import rospy
import message_filters
from pytictoc import TicToc
import numpy as np
from subprocess import Popen
import utils.utilities_ACMPFC as ACMPFC_utils
import time


def neural_networks_instantiation(num_ensembles, Device):
    # Instantiate the three neural networks models
    actor = ACMPFC_utils.NeuralNetwork(num_inputs=4,
                                      num_outputs=1,
                                      num_hidden_layers=2,
                                      num_hidden_neurons=300,
                                      dropout_prob=0).to(Device)

    # Critic
    critic = ACMPFC_utils.NeuralNetwork_Critic(num_inputs=1,
                                              num_outputs=1,
                                              num_hidden_layers=2,
                                              num_hidden_neurons=128,
                                              dropout_prob=0.1).to(Device)

    model_approximator = ACMPFC_utils.NeuralNetwork(num_inputs=4,
                                                   num_outputs=3,
                                                   num_hidden_layers=3,
                                                   num_hidden_neurons=300,
                                                   dropout_prob=0).to(device)
    return [actor, critic, model_approximator]


class ACMPFC():

    def __init__(self, actor, critic, model_approximator, device):
        self.iter = 0
        self.u_old = 0

        # Store NNs
        self.actor = actor

        self.critic = critic

        self.model_approximator = model_approximator

        self.device = device

        # Initialize ROS node and publishers/subscribers
        rospy.init_node('ACMPFC_online_actor', anonymous=False)

        def myhook():
            print("[ACMPFC] Shutting down...")

        self.initialize_subscribers_franka()
        self.initialize_subscribers_trainer()
        self.initialize_publishers()

        # Upload distribution params on ros server
        norm_data_path = 'norm_data/norm_data_example.csv'
        ACMPFC_utils.set_distribution_parameters(norm_data_path)
        time.sleep(1.)
        self.z_ee_dist, self.zdot_ee_dist, self.fz_ee_dist, self.error_f_dist, self.u_dist = ACMPFC_utils.get_distribution_parameters_lab(
        )
        ####
        u_u = 0 # EXAMPLE DATA, substitute if needed
        u_l = -0.05 # EXAMPLE DATA, substitute if needed

        mean_u = (u_u + u_l) / 2
        std_dev_u = (u_u - u_l) / 2
        #self.u_dist = (mean_u, std_dev_u) # Uncomment if you have your own environment (stiffness etc)

        mean_e_f = 0 # EXAMPLE DATA, substitute if needed
        std_dev_e_f = 13 # EXAMPLE DATA, substitute if needed
        #self.error_f_dist = (mean_e_f, std_dev_e_f) # Uncomment if you have your own environment (stiffness etc)

    # Instantiate the necessary subscribers and publishers to communicate with the simulation
    def initialize_subscribers_franka(self):
        
        self.cartesian_position_sub = message_filters.Subscriber(
            "/franka_ee_pose", PoseStamped, queue_size=1,
            buff_size=2**20) 
        self.cartesian_velocity_sub = message_filters.Subscriber("/franka_ee_velocity",
                                                                 TwistStamped,
                                                                 queue_size=1,
                                                                 buff_size=2**20)
        self.wrench_sub = message_filters.Subscriber("/franka_ee_wrench",
                                                     WrenchStamped,
                                                     queue_size=1,
                                                     buff_size=2**20)
        self.error_sub = message_filters.Subscriber("/error_f",
                                                    WrenchStamped,
                                                    queue_size=1,
                                                    buff_size=2**20)
        sync = message_filters.ApproximateTimeSynchronizer([
            self.cartesian_position_sub, self.cartesian_velocity_sub, self.wrench_sub,
            self.error_sub
        ],
                                                           queue_size=1,
                                                           slop=0.1)
        sync.registerCallback(self.topics_callback)

    def initialize_subscribers_trainer(self):
        self.actor_state_dict_sub = rospy.Subscriber("/actor_state_dict_path",
                                                     String,
                                                     callback=self.callback_state_dict,
                                                     callback_args=self.actor,
                                                     queue_size=10,
                                                     buff_size=2**20)

    def initialize_publishers(self):
        self.u_pub = rospy.Publisher("/u", PointStamped, queue_size=10)

    def topics_callback(self, pose, velocity, wrench, error_f_msg):
        # Start timing
        tictoc = TicToc()
        tictoc.tic()
        pos_z = pose.pose.position.z
        vel_z = velocity.twist.linear.z
        f_z = wrench.wrench.force.z
        error_f = error_f_msg.wrench.force.z

        # Normalize
        pos_z_norm = (pos_z - self.z_ee_dist[0]) / self.z_ee_dist[1]
        vel_z_norm = (vel_z - self.zdot_ee_dist[0]) / self.zdot_ee_dist[1]
        f_z_norm = (f_z - self.fz_ee_dist[0]) / self.fz_ee_dist[1]
        error_f_norm = (error_f - self.error_f_dist[0]) / self.error_f_dist[1]

        # Store state
        state_norm = np.array([pos_z_norm, vel_z_norm, f_z_norm])
        state_aug_norm = np.array([pos_z_norm, vel_z_norm, f_z_norm, error_f_norm])

        self.actor.eval()
        with torch.no_grad():
            action_norm = self.actor.forward(
                torch.from_numpy(state_aug_norm).to(
                    self.device).unsqueeze(0)).detach().cpu().numpy()
        u_msg = PointStamped()
        u_msg.point.x = action_norm[0][0] * self.u_dist[1] + self.u_dist[0] 

        delta = 1
        if self.u_old != 0:
            u_msg.point.x = np.clip(u_msg.point.x, self.u_old - delta,
                                    self.u_old + delta)

        self.u_old = u_msg.point.x

        self.u_pub.publish(u_msg)

        #tictoc.toc('ACMPFC control loop timing:') # Uncomment if you want to check the working frequency (mujoco is set to 500Hz, while in reality you can go higher)

    def callback_state_dict(self, path, model):
        # Retrieve the updated weights
        tictoc = TicToc()
        tictoc.tic()

        model.load_state_dict(torch.load(path.data))
        str = 'ACMPFC callback read ' + model._get_name() + ' timing:'


if __name__ == '__main__':
    device = 'cuda'
    actor, critic, ensemble = neural_networks_instantiation(num_ensembles=2,
                                                            Device=device)
    impedance_controller = ACMPFC(actor, critic, ensemble, device)
    print('[Controller] ACMPFC online controller running')

    # launch the offline trainer process for the neural networks
    Popen(['python3', 'ACMPFC_offline_trainer.py'])
    rospy.spin()
