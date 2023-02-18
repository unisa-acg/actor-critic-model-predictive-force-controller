import numpy as np
from numba import njit
import torch.nn as nn
import torch
from collections import OrderedDict
import rospy
import csv
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
import os


def read_data_from_csv(csv_file_path):
    """Reads the data from .csv files

    Args:
        csv_file_path: path of the .csv file whose data will be read
    """
    fieldnames = next(csv.reader(open(csv_file_path), delimiter=","))
    robot_info = np.loadtxt(csv_file_path, dtype=np.float64, delimiter=",", skiprows=1)

    return [fieldnames, robot_info]


def create_dataloader(x, y, batch_size=64, shuffle=True):
    """Creates a Dataloader instance given two arrays x and y

    Args:
        x (NDArray): x data
        y (NDArray): y_data
        batch_size (int, optional): Defaults to 64.
        shuffle (bool, optional): Shuffle data. Defaults to True.

    Returns:
        DataLoader
    """
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
    return dataloader


def set_distribution_parameters(norm_data_path):
    fieldnames, norm_data = read_data_from_csv(norm_data_path)
    distribution_dict = {}
    for n in range(len(fieldnames)):
        distribution_dict[fieldnames[n]] = (float(norm_data[0,
                                                            n]), float(norm_data[1, n]))
    rospy.set_param('distribution_dict', distribution_dict)


def get_distribution_parameters_lab():
    distribution_dict = rospy.get_param('distribution_dict')
    z_ee_dist = distribution_dict['pos_z']
    zdot_ee_dist = distribution_dict['vel_z']
    fz_ee_dist = distribution_dict['f_z']
    x_f_dist = distribution_dict['x_f']
    return tuple(np.float32(z_ee_dist)), tuple(np.float32(zdot_ee_dist)), tuple(
        np.float32(fz_ee_dist)), tuple(np.float32(x_f_dist))


def get_distribution_parameters():
    distribution_dict = rospy.get_param('distribution_dict')
    z_ee_dist = distribution_dict['z_ee_t']
    zdot_ee_dist = distribution_dict['zdot_ee_t']
    fz_ee_dist = distribution_dict['f_z_ee_t']
    error_f_dist = distribution_dict['e_f']
    u_dist = distribution_dict['u_t']
    return tuple(np.float32(z_ee_dist)), tuple(np.float32(zdot_ee_dist)), tuple(
        np.float32(fz_ee_dist)), tuple(np.float32(error_f_dist)), tuple(
            np.float32(u_dist))


# ----------------------------- CRITIC FUNCTIONS ----------------------------- #


class NeuralNetwork_Critic(nn.Module):
    """Creates a NN class with the desired dimensions and, optionally, dropout layers
    (by default disabled)"""

    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_hidden_neurons,
        num_hidden_layers,
        dropout_prob=1,
    ):
        super(NeuralNetwork_Critic, self).__init__()
        self.flatten = nn.Flatten()
        self.input_dim = num_inputs
        self.output_dim = num_outputs
        self.hidden_dim = num_hidden_neurons
        self.layers = OrderedDict()

        # Add input layer
        self.layers["lin" + str(0)] = nn.Linear(self.input_dim, self.hidden_dim)
        self.layers["relu" + str(0)] = nn.ReLU()

        # Add hidden layers with dropout possibility
        for i in range(1, num_hidden_layers + 1):
            if dropout_prob != 1:
                if 0 <= dropout_prob <= 1:
                    self.layers["drop" + str(i)] = nn.Dropout(p=dropout_prob)
                else:
                    raise Exception("Dropout probability must be between 0 and 1")
            self.layers["lin" + str(i)] = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.layers["relu" + str(i)] = nn.ReLU()

        # Add output layer
        self.layers["lin" + str(num_hidden_layers + 2)] = nn.Linear(
            self.hidden_dim, self.output_dim)
        self.pipe = nn.Sequential(self.layers)

    def forward(self, x):
        x = self.flatten(x.float())
        y_predicted = self.pipe(x)
        return torch.abs(y_predicted)


def Q_interp_critic(critic, range_interp=(-50, 50), device='cpu'):
    critic.eval()
    SIZEN = 1000
    AAA = np.linspace(range_interp[0], range_interp[1], SIZEN,
                      dtype=np.float32) * np.ones([1, SIZEN], dtype=np.float32)
    BBB = np.zeros([1, SIZEN], dtype=np.float32)

    for itera in range(0, SIZEN):
        critic.eval()
        InputAA = torch.from_numpy(AAA[:, itera]).unsqueeze(1).to(device)
        with torch.no_grad():
            BBB[:, itera] = (critic.forward(InputAA).detach().cpu().numpy())
    Q_interp = np.polyfit(AAA[0, :], BBB[0, :], 2)

    Q_y = Q_interp[0] * AAA[0, :]**2 + Q_interp[1] * AAA[0, :]**1 + Q_interp[2]
    idx_max = np.argmax(Q_y)

    return Q_interp


class NeuralNetwork(nn.Module):
    """Creates a NN class with the desired dimensions and, optionally, dropout layers
    (by default disabled)"""

    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_hidden_neurons,
        num_hidden_layers,
        dropout_prob=1,
    ):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.input_dim = num_inputs
        self.output_dim = num_outputs
        self.hidden_dim = num_hidden_neurons
        self.layers = OrderedDict()

        # Add input layer
        self.layers["lin" + str(0)] = nn.Linear(self.input_dim, self.hidden_dim)
        self.layers["relu" + str(0)] = nn.ReLU()

        # Add hidden layers with dropout possibility
        for i in range(1, num_hidden_layers + 1):
            if dropout_prob != 1:
                if 0 <= dropout_prob <= 1:
                    self.layers["drop" + str(i)] = nn.Dropout(p=dropout_prob)
                else:
                    raise Exception("Dropout probability must be between 0 and 1")
            self.layers["lin" + str(i)] = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.layers["relu" + str(i)] = nn.ReLU()

        # Add output layer
        self.layers["lin" + str(num_hidden_layers + 2)] = nn.Linear(
            self.hidden_dim, self.output_dim)
        self.pipe = nn.Sequential(self.layers)

    def forward(self, x):
        x = self.flatten(x.float())
        y_predicted = self.pipe(x)
        return y_predicted


def modify_XML():
    tree = ET.parse('sim_model.xml')
    root = tree.getroot()

    for body in root.iter('body'):
        if body.attrib['name'] == 'gripper0_eef':
            attrib_geom = {
                'name': 'ee_sphere_visual',
                'size': '0.4 0.4 0.4',
                'type': "sphere",
                'solref': "0.01 1",
                'contype': "0",
                'conaffinity': "0",
                'group': "1",
                'rgba': "1 1 1 1",
                'mesh': "robot0_link7_vis"
            }
            geom_to_add = ET.Element('geom', attrib_geom)
            attrib_geom2 = {
                'name': 'ee_sphere_collision',
                'size': '0.4 0.4 0.4',
                'type': "sphere",
                'rgba': "1 1 1 1",
                'mesh': "robot0_link7"
            }
            geom2_to_add = ET.Element('geom', attrib_geom2)
            attrib_body = {'name': 'ee_sphere', 'pos': '-0.01 -0.009 -0.07'}
            body_to_add = ET.Element('body', attrib_body)
            body_to_add.append(geom_to_add)
            body_to_add.append(geom2_to_add)
            body.append(body_to_add)

    xml_string = ET.tostring(root, encoding='utf8').decode('utf8')
    tree.write('sim_model_mod.xml')

    return xml_string


class ReplayBuffer():

    def __init__(self, state_dim=3, state_aug_dim=4, action_dim=1, mem_size=50000):
        super(ReplayBuffer, self).__init__()
        self.mem_size = mem_size
        self.mem_counter = 0
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.state_aug_dim = state_aug_dim
        self.reset_storage()
        self.episodes = []

    def store_transition(self, state, state_aug, action, done):
        idx = self.mem_counter
        self.state_buffer[idx] = state
        self.state_aug_buffer[idx] = state_aug
        self.action_buffer[idx] = action

        self.done_buffer[idx] = done

        if done == 1:
            self.next_state_buffer[0:idx - 1] = self.state_buffer[1:idx]
            self.store_episode(idx - 1)
            self.reset_storage()
            self.mem_counter = 0
        else:
            self.mem_counter += 1

    def store_episode(self, terminal_idx):

        state_hist = self.state_buffer[0:terminal_idx]
        state_aug_hist = self.state_aug_buffer[0:terminal_idx]
        action_hist = self.action_buffer[0:terminal_idx]
        next_state_hist = self.next_state_buffer[0:terminal_idx]
        done_hist = self.done_buffer[0:terminal_idx]
        cem_temp = self.done_buffer[0:terminal_idx]
        episode_len = terminal_idx
        episode = Episode(state_hist, state_aug_hist, action_hist, next_state_hist,
                          done_hist, cem_temp, episode_len)
        self.episodes.append(episode)

    def reset_storage(self):
        self.state_buffer = np.zeros((self.mem_size, self.state_dim), dtype=np.float32)
        self.state_aug_buffer = np.zeros((self.mem_size, self.state_aug_dim),
                                         dtype=np.float32)
        self.next_state_buffer = np.zeros((self.mem_size, self.state_dim),
                                          dtype=np.float32)
        self.action_buffer = np.zeros((self.mem_size, self.action_dim),
                                      dtype=np.float32)
        self.done_buffer = np.zeros((self.mem_size, 1), dtype=np.float32)

    def get_last_episode_data(self, max_ep_length=5000):
        last_ep = self.episodes[-1]
        if last_ep.episode_len > max_ep_length:
            idx = max_ep_length
        else:
            idx = last_ep.episode_len

        states = last_ep.state_hist[0:idx, :]
        states_aug = last_ep.state_aug_hist[0:idx, :]
        actions = last_ep.action_hist[0:idx, :]
        next_states = last_ep.next_state_hist[0:idx, :]
        return states, states_aug, actions, next_states

    def get_last_CEM_data(self, max_ep_length=5000):
        last_ep = self.episodes[-1]
        if last_ep.episode_len > max_ep_length:
            idx = max_ep_length
        else:
            idx = last_ep.episode_len

        return last_ep.cem_hist

    def get_last_N_episodes_data(self, n_episodes=2, max_ep_length=5000):

        episodes_available = len(self.episodes)
        if n_episodes > episodes_available:
            n_episodes = episodes_available

        for i in range(0, n_episodes):
            episode_n = self.episodes[-(i + 1)]
            if episode_n.episode_len > max_ep_length:
                idx = max_ep_length
            else:
                idx = episode_n.episode_len

            states_n_ep = episode_n.state_hist[0:idx, :]
            states_aug_n_ep = episode_n.state_aug_hist[0:idx, :]
            actions_n_ep = episode_n.action_hist[0:idx, :]
            next_states_n_ep = episode_n.next_state_hist[0:idx, :]

            if i == 0:
                states = states_n_ep
                states_aug = states_aug_n_ep
                actions = actions_n_ep
                next_states = next_states_n_ep
            else:
                states = np.concatenate((states, states_n_ep), axis=0)
                states_aug = np.concatenate((states_aug, states_aug_n_ep), axis=0)
                actions = np.concatenate((actions, actions_n_ep), axis=0)
                next_states = np.concatenate((next_states, next_states_n_ep), axis=0)

        return states, states_aug, actions, next_states

    def get_random_dataloader_from_history(self, n_samples=500):
        pass

    def write_episode_to_csv(self, path_episodes_save):
        if not os.path.isdir(path_episodes_save):
            os.makedirs(path_episodes_save)
        states, states_aug, actions, next_states = self.get_last_episode_data(
            max_ep_length=15000)
        cem = self.get_last_CEM_data(max_ep_length=15000)
        episode_counter = len(self.episodes)
        episode_str = 'Episode_' + str(episode_counter) + '.csv'
        # data_matrix = np.concatenate((states, actions, np.expand_dims(cem, axis=-1)),
        #                              axis=1)
        data_matrix = np.concatenate((states_aug, actions, cem), axis=1)
        with open(os.path.join(path_episodes_save, episode_str), 'w') as f:
            write = csv.writer(f)
            header = ['z_ee_t', 'zdot_ee_t', 'f_z_ee_t', 'e_f', 'u', 'u_cem']
            write.writerow(header)
            write.writerows(data_matrix)

    def store_CEM(self, action_CEM, n_episodes):
        last_ep_len = self.episodes[-1].episode_len
        # last_CEM = action_CEM[:last_ep_len]
        last_CEM = action_CEM[-last_ep_len:]
        self.episodes[-1].cem_hist = last_CEM


@dataclass
class Episode:
    state_hist: np.ndarray
    state_aug_hist: np.ndarray
    action_hist: np.ndarray
    next_state_hist: np.ndarray
    done_hist: np.ndarray
    cem_hist: np.ndarray
    episode_len: int