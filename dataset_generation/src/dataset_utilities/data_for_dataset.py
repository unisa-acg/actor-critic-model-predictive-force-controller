import csv
import os.path
import dataset_generation.src.utilities.quat_utils as quat_utils
import numpy as np


class DataForDataset:
    """
    Class that contains method to plot and store in .csv files the info happening during 
    the MuJoCo simulation for the creation of the dataset. 
    The data from the simulation are stored in a folder selected by the user.
    """

    def __init__(self, steps):
        """Instantiate the variables need for the other methods.

        Args:
            steps: number of total simulation steps 
        """
        self.ee_pos = np.zeros((steps, 3), dtype=np.float64, order="C")
        self.ee_ori = np.zeros((steps, 3), dtype=np.float64, order="C")
        self.ee_vel = np.zeros((steps, 3), dtype=np.float64, order="C")
        self.ee_omg = np.zeros((steps, 3), dtype=np.float64, order="C")
        self.ee_force = np.zeros((steps, 3), dtype=np.float64, order="C")
        self.ee_torque = np.zeros((steps, 3), dtype=np.float64, order="C")

        self.ncalls_get_info_robot = 0

    def get_info_robot(self, env, bodies_name):
        """It extracts the useful data from the simulation

        Args:
            env: MuJoCo environment from which allows access to sim (mjSim -> MuJoCo 
            simulation instance at current timestep)
            bodies_name: name of the bodies whose contact data need to be obtained 
        """
        self.ee_force[self.ncalls_get_info_robot, :] = env.robots[0].ee_force

        self.ee_torque[self.ncalls_get_info_robot, :] = env.robots[0].ee_torque

        self.ee_pos[self.ncalls_get_info_robot, :] = env.sim.data.site_xpos[
            env.sim.model.site_name2id(bodies_name)]

        self.ee_vel[self.ncalls_get_info_robot, :] = env.sim.data.site_xvelp[
            env.sim.model.site_name2id(bodies_name)]

        self.ee_omg[self.ncalls_get_info_robot, :] = env.sim.data.site_xvelr[
            env.sim.model.site_name2id(bodies_name)]

        self.ee_quat = np.array(
            env.sim.data.site_xmat[env.sim.model.site_name2id(bodies_name)].reshape(
                [3, 3]))

        self.ee_ori[self.ncalls_get_info_robot, :] = quat_utils.mat2euler(self.ee_quat)

        self.ncalls_get_info_robot += 1

    def contact_info_to_csv(self, csv_name, csv_dir_path):
        """After calling get_info_robot() method, prints the contact data into .csv 
        files in the folder csv_folder

        Args:
            csv_file_path: path of the .csv file where the data will be saved
        """

        csv_full_path = os.path.join(csv_dir_path, csv_name)

        # Check if directory exists, otherwise create it
        if not os.path.isdir(csv_dir_path):
            os.makedirs(csv_dir_path)

        pos = ['pos_x', 'pos_y', 'pos_z']
        ori = ['ori_x', 'ori_y', 'ori_z']
        vel = ['vel_x', 'vel_y', 'vel_z']
        omg = ['omg_x', 'omg_y', 'omg_z']
        delta_pos = ['delta_pos_x', 'delta_pos_y', 'delta_pos_z']
        delta_ori = ['delta_ori_x', 'delta_ori_y', 'delta_ori_z']
        force = ['f_x', 'f_y', 'f_z']
        torque = ['torque_x', 'torque_y', 'torque_z']
        force_output = ['f_x_out', 'f_y_out', 'f_z_out']
        torque_output = ['torque_x_out', 'torque_y_out', 'torque_z_out']

        fieldnames = (pos + ori + vel + omg + delta_pos + delta_ori + force_output +
                      torque_output)

        force = np.stack(
            (
                np.array(self.ee_force[:, 1], dtype=np.float64),
                np.array(self.ee_force[:, 0], dtype=np.float64),
                np.array(self.ee_force[:, 2], dtype=np.float64),
            ),
            axis=-1,
        )

        delta_pos = np.array([
            np.subtract(self.ee_pos[i], self.ee_pos[i + 1])
            for i in range(len(self.ee_pos) - 1)
        ],
                             dtype=np.float64)
        delta_ori = np.array([
            np.subtract(self.ee_ori[i], self.ee_ori[i + 1])
            for i in range(len(self.ee_ori) - 1)
        ],
                             dtype=np.float64)

        self.robot_info = np.concatenate((
            self.ee_pos[:self.ncalls_get_info_robot - 1, :],
            self.ee_ori[:self.ncalls_get_info_robot - 1, :],
            self.ee_vel[:self.ncalls_get_info_robot - 1, :],
            self.ee_omg[:self.ncalls_get_info_robot - 1, :],
            delta_pos[:self.ncalls_get_info_robot - 1, :],
            delta_ori[:self.ncalls_get_info_robot - 1, :],
            force[1:self.ncalls_get_info_robot, :],
            self.ee_torque[1:self.ncalls_get_info_robot, :],
        ),
                                         axis=1)

        with open(csv_full_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            writer.writerows(self.robot_info)
