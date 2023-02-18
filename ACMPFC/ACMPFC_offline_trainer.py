import torch.nn as nn
import torch
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped, PointStamped
from std_msgs.msg import String, Bool
import rospy
import message_filters
from pytictoc import TicToc
import numpy as np
import os
import time
import utils.utilities_ACMPFC as ACMPFC_utils
from utils.utilities_ACMPFC import ReplayBuffer, create_dataloader
from torch.utils.tensorboard import SummaryWriter

# Write data to log
writer = SummaryWriter()


def neural_networks_instantiation(num_ensembles, Device):
    # Actor

    actor = ACMPFC_utils.NeuralNetwork(
        num_inputs=4,
        num_outputs=1,
        num_hidden_layers=2,  #2
        num_hidden_neurons=300,
        dropout_prob=0).to(Device)

    # Critic
    critic = ACMPFC_utils.NeuralNetwork_Critic(num_inputs=2,
                                              num_outputs=1,
                                              num_hidden_layers=2,
                                              num_hidden_neurons=128,
                                              dropout_prob=0.1).to(Device)

    # critic = ACMPFC_utils.NeuralNetwork_Critic(num_inputs=6,
    #                                           num_outputs=1,
    #                                           num_hidden_layers=2,
    #                                           num_hidden_neurons=128,
    #                                           dropout_prob=0.1).to(Device)

    model_approximator = ACMPFC_utils.NeuralNetwork(num_inputs=4,
                                                   num_outputs=3,
                                                   num_hidden_layers=3,
                                                   num_hidden_neurons=300,
                                                   dropout_prob=0).to(Device)

    return [actor, critic, model_approximator]


class Trainer():

    def __init__(self, actor, critic, model_approximator, device):
        self.iter = 0
        self.iter_batch = 0
        self.iter_batch_actor = 0
        self.iter_batch_critic = 0
        self.counter_debug = 0
        self.iter_counter = 0
        self.counter_epochs = 0
        self.counter_sim = 0
        self.cem_counter = 0
        # Store NNs
        self.actor = actor
        self.actor_loss = nn.MSELoss()
        self.learning_rate_actor = 1e-4
        self.actor_optim = torch.optim.Adam(self.actor.parameters(),
                                            lr=self.learning_rate_actor,
                                            weight_decay=4e-5)  # best 1 e-4 and 4e-5

        self.critic = critic
        self.critic_loss = nn.MSELoss()  # nn.HuberLoss()
        # self.critic_optim = torch.optim.SGD(self.critic.parameters(),
        #                                     lr=1e-3,
        #                                     momentum=0.9)  #weight_decay=4e-5)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(),
                                             lr=1e-4,
                                             weight_decay=4e-5)
        self.critic_training = False
        self.critic_load_dict = False
        if self.critic_load_dict:
            self.critic.load_state_dict(
                torch.load('nn_state_dict_save_no_time/critic.pth'))

            print('Critic state dict loaded!')

        self.actor_training = True
        self.actor_load_dict = False
        if self.actor_load_dict:
            self.actor.load_state_dict(
                torch.load('nn_state_dict_save_no_time/actor.pth'))
            print('Actor state dict loaded!')

        self.model_approximator = model_approximator
        self.model_approximator_loss = nn.MSELoss()
        self.model_approximator_optim = torch.optim.Adam(
            self.model_approximator.parameters(), lr=1e-4, weight_decay=4e-5)

        self.m_a_training = True
        self.m_a_load_dict = False
        if self.m_a_load_dict:
            self.model_approximator.load_state_dict(
                torch.load('nn_state_dict_save_no_time/model.pth'))
            print('Model approx state dict loaded!')

        self.device = device

        self.RB = ReplayBuffer(state_dim=3,
                               state_aug_dim=4,
                               action_dim=1,
                               mem_size=100000)

        # Initialize ROS node and publishers/subscribers
        rospy.init_node('Trainer', anonymous=False)

        def myhook():
            print("[Trainer] Trainer shutting down...")

        rospy.on_shutdown(myhook)
        self.initialize_subscribers()
        self.initialize_publishers()

        self.z_ee_dist, self.zdot_ee_dist, self.fz_ee_dist, self.error_f_dist, self.u_dist = ACMPFC_utils.get_distribution_parameters_lab(
        )

        u_u = 0 # EXAMPLE DATA, substitute if needed
        u_l = -0.05  # EXAMPLE DATA, substitute if needed

        mean_u = (u_u + u_l) / 2
        std_dev_u = (u_u - u_l) / 2
        self.u_dist = (mean_u, std_dev_u)

        mean_e_f = 0 # EXAMPLE DATA, substitute if needed
        std_dev_e_f = 13 # EXAMPLE DATA, substitute if needed
        self.error_f_dist = (mean_e_f, std_dev_e_f)

    def initialize_publishers(self):
        self.received_pub = rospy.Publisher("/done_received", Bool, queue_size=10)
        self.done_training_pub = rospy.Publisher("/done_training", Bool, queue_size=10)
        #Define the state dictionaries path publishers
        self.m_a_state_dict_pub = rospy.Publisher("/m_a_state_dict_path",
                                                  String,
                                                  queue_size=1)
        self.critic_state_dict_pub = rospy.Publisher("/critic_state_dict_path",
                                                     String,
                                                     queue_size=1)
        self.actor_state_dict_pub = rospy.Publisher("/actor_state_dict_path",
                                                    String,
                                                    queue_size=1)

        # Define the messages
        self.actor_sd_msg = String()
        self.critic_sd_msg = String()
        self.m_a_sd_msg = String()

        # Create the folder to store state dict
        cwd = os.getcwd()
        main_save_folder_path = os.path.join(cwd, 'nn_state_dict_save_no_time')

        if os.path.exists(main_save_folder_path) is False:
            os.mkdir(main_save_folder_path)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.path_folders_saves = main_save_folder_path

    def initialize_subscribers(self):
        self.cartesian_position_sub = message_filters.Subscriber(
            "/franka_ee_pose", PoseStamped, queue_size=10,
            buff_size=2**20)  # buffer size = 10, queue = 1
        self.cartesian_velocity_sub = message_filters.Subscriber("/franka_ee_velocity",
                                                                 TwistStamped,
                                                                 queue_size=10,
                                                                 buff_size=2**20)
        self.wrench_sub = message_filters.Subscriber("/franka_ee_wrench",
                                                     WrenchStamped,
                                                     queue_size=10,
                                                     buff_size=2**20)
        self.error_sub = message_filters.Subscriber("/error_f",
                                                    WrenchStamped,
                                                    queue_size=10,
                                                    buff_size=2**20)
        self.u_sub = message_filters.Subscriber("/u",
                                                PointStamped,
                                                queue_size=10,
                                                buff_size=2**20)
        self.done_sub = message_filters.Subscriber("/done",
                                                   PointStamped,
                                                   queue_size=10,
                                                   buff_size=2**20)
        sync = message_filters.ApproximateTimeSynchronizer([
            self.cartesian_position_sub, self.cartesian_velocity_sub, self.wrench_sub,
            self.error_sub, self.u_sub, self.done_sub
        ],
                                                           queue_size=10,
                                                           slop=0.1)
        sync.registerCallback(self.data_callback)

    def data_callback(self, pose, velocity, wrench, error_f_msg, u_msg, done_msg):
        pos_z = pose.pose.position.z
        vel_z = velocity.twist.linear.z
        f_z = wrench.wrench.force.z
        error_f = error_f_msg.wrench.force.z
        u = u_msg.point.x
        done = done_msg.point.x

        # Normalize
        pos_z_norm = (pos_z - self.z_ee_dist[0]) / self.z_ee_dist[1]
        vel_z_norm = (vel_z - self.zdot_ee_dist[0]) / self.zdot_ee_dist[1]
        f_z_norm = (f_z - self.fz_ee_dist[0]) / self.fz_ee_dist[1]
        error_f_norm = (error_f - self.error_f_dist[0]) / self.error_f_dist[1]
        u_norm = (u - self.u_dist[0]) / self.u_dist[1]

        # Store state 
        state_norm = np.array([pos_z_norm, vel_z_norm, f_z_norm])
        state_aug_norm = np.array([pos_z_norm, vel_z_norm, f_z_norm, error_f_norm])
        action_norm = u_norm

        # If done, start training of last episode, 
        # else store one every five transition (100 Hz frequency of storage if Mujoco set to 500 Hz)
        if done == 1:
            self.publish_received()
            self.RB.store_transition(state_norm, state_aug_norm, action_norm, done)
            self.RB.write_episode_to_csv(path_episodes_save='csv_episode_storage')
            self.iter_counter = 0
            print('[Trainer] Received Done, start replay training..')
            self.train()
            self.publish_nns_state_dict_paths()
            self.publish_done_training()
        else:
            if self.iter_counter == 5:
                self.RB.store_transition(state_norm, state_aug_norm, action_norm, done)
                self.iter_counter = 0
            else:
                self.iter_counter += 1

    def publish_received(self):
        # Send the command to close the simulation
        received_msg = Bool()
        received_msg.data = True
        self.received_pub.publish(received_msg)

    def train(self):
        T = TicToc()
        T.tic()
        # Retrieve the data from last episode(s)
        states, states_aug, actions, next_states = self.RB.get_last_episode_data(
            max_ep_length=8000)
        states_n_ep, states_aug_n_ep, actions_n_ep, next_states_n_ep = self.RB.get_last_N_episodes_data(
            n_episodes=3, max_ep_length=80000)

        states_n_ep_critic, states_aug_n_ep_critic, actions_n_ep_critic, next_states_n_ep_critic = self.RB.get_last_N_episodes_data(
            n_episodes=5, max_ep_length=80000)

        #### MODEL APPROXIMATOR TRAINING
        if self.m_a_training == True:

            x_train_ma = np.concatenate((states_n_ep, actions_n_ep), axis=1)
            y_train_ma = np.array(next_states_n_ep, dtype=np.float32)
            train_loader_ma = create_dataloader(
                torch.from_numpy(x_train_ma).to(self.device).float(),
                torch.from_numpy(y_train_ma).to(self.device).float(),
                shuffle=True,
            )
            N_EPOCHS = 50  # 200 lab
            self.model_approximator.train()
            for _ in range(N_EPOCHS):
                for id_batch, (x_batch_ma, y_batch_ma) in enumerate(train_loader_ma):
                    self.model_approximator_optim.zero_grad()
                    action_NN_ma = self.model_approximator.forward(x_batch_ma)
                    loss_ma = self.model_approximator_loss(action_NN_ma, y_batch_ma)
                    loss_ma.backward()
                    self.model_approximator_optim.step()
                    writer.add_scalar("Loss/train_model_approx", loss_ma,
                                      self.iter_batch)
                    #print('loss model approximator: ', loss_ma)
                    self.iter_batch += 1

            T.toc('[Trainer] DONE Training model approximator finished in')

        #### CRITIC TRAIN
        if self.critic_training == True:
            states_aug_critic = np.expand_dims(states_aug_n_ep_critic[:, 3], axis=1)
            actions_critic = actions_n_ep_critic
            input_critic = np.concatenate((states_aug_critic, actions_critic), axis=1)
            input_critic_torch = torch.from_numpy(input_critic).to(self.device)

            self.critic.train()
            N_EPOCHS = 5  #5
            DELTA = 1
            GAMMA = 0.95
            for _ in range(N_EPOCHS):
                for i in range(1, len(states_aug_critic) - 1):
                    Q_n = self.critic.forward(input_critic_torch[i, :].unsqueeze(0))
                    Q_np1 = self.critic.forward(input_critic_torch[i +
                                                                   1, :].unsqueeze(0))

                    term_error_f = abs(input_critic_torch[
                        i + 1,
                        0])
                    term_action1 = (input_critic_torch[i, 1])**2
                    term_action2 = abs(input_critic_torch[i, 1] -
                                       input_critic_torch[i - 1, 1])
                    c0 = 100
                    c1 = 0
                    c2 = 0  #50  #600
                    r = c0 * term_error_f + c1 * term_action1 + c2 * term_action2
                    writer.add_scalar('Cost/term_force', term_error_f * c0, i)
                    writer.add_scalar('Cost/term_action1', term_action1 * c1, i)
                    writer.add_scalar('Cost/term_action2', term_action2 * c2, i)
                    Q_bman = r + GAMMA * Q_np1  
                    loss = self.critic_loss(Q_n, Q_bman)
                    loss.backward(retain_graph=True)
                    self.critic_optim.step()
                    if i % 64 == 0:
                        writer.add_scalar("Loss/train_critic", loss,
                                          self.iter_batch_critic)
                        self.iter_batch_critic += 1
            T.toc('[Trainer] DONE Training critic finished in')
            self.counter_sim += 1

        ###ACTOR TRAINING
        if self.actor_training == True:
            x_train = np.array(states_aug_n_ep_critic)
            y = self.CEM_critic(states_n_ep_critic, states_aug_n_ep_critic)
            action_CEM = (np.array(y, dtype=np.float32))

            self.RB.store_CEM(action_CEM, 1)

            y_train = np.expand_dims(action_CEM, axis=-1)
            train_loader = create_dataloader(
                torch.from_numpy(x_train).to(self.device).float(),
                torch.from_numpy(y_train).to(self.device).float(),
                shuffle=True,
            )
            N_EPOCHS = 50  #50 lab #5
            lambda1 = lambda epoch: 0.7**epoch
            self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(self.actor_optim,
                                                                     lr_lambda=lambda1)
            self.actor.train()
            self.actor_optim.param_groups[0]["lr"] = self.learning_rate_actor
            for epoch in range(N_EPOCHS):
                for id_batch, (x_batch, y_batch) in enumerate(train_loader):
                    self.actor_optim.zero_grad()
                    action_NN = self.actor.forward(x_batch)
                    loss = self.actor_loss(action_NN, y_batch)
                    loss.backward()  #(retain_graph=True)
                    self.actor_optim.step()
                    writer.add_scalar("Loss/train_actor", loss, self.iter_batch_actor)
                    #print('loss actor: ', loss)
                    self.iter_batch_actor += 1

                # Incremental epochs if target loss not reached
                if loss > 2: # EXAMPLE DATA, change if needed
                    if N_EPOCHS < 20:
                        N_EPOCHS += 1

                writer.add_scalar("Loss/train_actor_lr",
                                  self.actor_optim.param_groups[0]["lr"],
                                  self.counter_epochs)
                #self.actor_scheduler.step() # Uncomment if needed in real case scenario
                #self.actor_optim.param_groups[0]["lr"] = self.#actor_optim.param_groups[0][
                #    "lr"] * 0.7  # actual best 0.9 con 1e-3 di partenza
                #print(self.actor_optim.param_groups[0]["lr"])
                self.counter_epochs += 1
            T.toc('[Trainer] DONE Training actor finished in')
        if self.cem_counter < 3:
            self.cem_counter += 1
        writer.flush()

    def publish_done_training(self):
        done_training_msg = Bool()
        done_training_msg.data = True
        self.done_training_pub.publish(done_training_msg)

    def unison_shuffled_copies(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def CEM_critic(self, state_n_norm, state_n_norm_aug):
        # State action details
        num_states = state_n_norm_aug.shape[0]

        # CEM details
        smoothing_rate = 0.95
        iterations = 3  
        num_elites = 4
        num_samples = 256
        time_horizon = 5 
        best_actions = []
      
        # Threshold
        ll_u = -1  
        ul_u = 1  

        for i in range(num_states):

            # Initializing:
            mu_matrix_u = np.zeros((1, time_horizon))
            std_matrix_u = np.ones((1, time_horizon))

            for _ in range(iterations):

                reward_sum = np.zeros((num_samples, 1))
                
                # Draw random samples from a normal (Gaussian) distribution
                u_samples = np.random.normal(loc=mu_matrix_u,
                                             scale=std_matrix_u,
                                             size=(num_samples, time_horizon))


                u_samples[u_samples >= ul_u] = ul_u
                u_samples[u_samples <= ll_u] = ll_u

                state_norm_tiled = torch.from_numpy(
                    np.tile(state_n_norm[i, :], (num_samples, 1))).to(self.device)

                f_d = (state_n_norm_aug[i, 2] * self.fz_ee_dist[1] + self.fz_ee_dist[0]
                      ) - (state_n_norm_aug[i, 3] * self.error_f_dist[1] +
                           self.error_f_dist[0])

                u_samples_torch = torch.from_numpy(u_samples).to(self.device)

                for t in range(time_horizon):
                    if t == 0:
                        state_action_norm_t = torch.cat(
                            (state_norm_tiled, (u_samples_torch[:, t]).unsqueeze(1)),
                            axis=1)
                    else:
                        state_action_norm_t = torch.cat(
                            (state_tp1, (u_samples_torch[:, t]).unsqueeze(1)), axis=1)
                    self.model_approximator.eval()
                    with torch.no_grad():
                        state_tp1 = self.model_approximator.forward(state_action_norm_t)
                        state_tp1_error_force_not_norm = (
                            state_tp1[:, 2] * self.fz_ee_dist[1] +
                            self.fz_ee_dist[0]) - f_d

                        state_tp1_error_force_norm = (
                            state_tp1_error_force_not_norm -
                            self.error_f_dist[0]) / self.error_f_dist[1]
                        c_error = 1
                        action = u_samples_torch[:, t].detach().cpu().numpy()
                        c_state = (10-self.cem_counter)/10
                        z_next = state_tp1[:,0].detach().cpu().numpy()
                        z = state_action_norm_t[:,0].detach().cpu().numpy()
                        reward = c_error * abs(state_tp1_error_force_norm.detach().cpu().numpy()) + c_state * abs(z_next - z)
                        action_tm1 = action
                    reward_sum = np.add(reward_sum, np.expand_dims(reward, axis=1))

                # For min
                reward_sum_list = list(np.squeeze(reward_sum))
                elites_idx = sorted(range(len(reward_sum_list)),
                                    key=lambda i: reward_sum_list[i])[:num_elites]

                elites_costs = reward_sum[elites_idx] 

                elites_u = u_samples[elites_idx, :]
                mu_matrix_u_new = np.sum(elites_u, axis=0) / num_elites
                std_matrix_u_new = np.sqrt(np.sum(np.square(elites_u -
                                                            mu_matrix_u_new)))

                # Update distribution of matrix u and D
                mu_matrix_u = smoothing_rate * mu_matrix_u_new + (
                    1 - smoothing_rate) * mu_matrix_u
                std_matrix_u = smoothing_rate * std_matrix_u_new + (
                    1 - smoothing_rate) * std_matrix_u

            best_actions.append(elites_u[0, 0])

        return best_actions

    def publish_nns_state_dict_paths(self):

        # Model Approximator
        PATH_model_approximator = os.path.join(self.path_folders_saves, 'model.pth')
        torch.save(self.model_approximator.state_dict(), PATH_model_approximator)

        # Critic
        PATH_critic = os.path.join(self.path_folders_saves, 'critic.pth')
        torch.save(self.critic.state_dict(), PATH_critic)

        # Actor
        PATH_actor = os.path.join(self.path_folders_saves, 'actor.pth')
        torch.save(self.actor.state_dict(), PATH_actor)

        # Copy paths on ROS messages
        self.m_a_sd_msg.data = PATH_model_approximator
        self.critic_sd_msg.data = PATH_critic
        self.actor_sd_msg.data = PATH_actor

        # Publish the messages
        self.m_a_state_dict_pub.publish(self.m_a_sd_msg.data)
        self.critic_state_dict_pub.publish(self.critic_sd_msg.data)
        self.actor_state_dict_pub.publish(self.actor_sd_msg.data)


if __name__ == '__main__':
    device = 'cuda'
    actor, critic, ensemble = neural_networks_instantiation(num_ensembles=3,
                                                            Device=device) # For simulation, due to low noise an ensemble can be superflous 
    trainer = Trainer(actor, critic, ensemble, device)
    print('[Trainer] Trainer running')
    rospy.spin()