import csv
import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob


class DataProcessing:
    """
    Class that contains method to manipulate the data stored in the folder 'csv_folder' (normalization and Gaussian noise addition) and store them in .csv files.
    This dataset will be used to train the neural network whose task is to learn the forces produced by the robot-environment contact.

    Methods
    -------
    * read_data_from_csv(env, csv_file_path): it reads the data from a .csv file, saving the header and the data

    * contact_info_processed_to_csv(n): after calling read_data_from_csv() method, it processes (normalization and Gaussian noise addition) all the contact information registered during the simulation and stores them into a .csv file in the folder 'csv_folder'

    * plot_data_comparison(): after calling read_data_from_csv() and contact_info_processed_to_csv() method, plot the info registered vs the info processed
    """

    def read_data_from_csv(self, csv_file_path):
        """It reads the data from .csv files

        Args:
            csv_file_path: path of the .csv file whose data will be red
        """
        self.fieldnames = next(csv.reader(open(csv_file_path), delimiter=','))
        self.robot_info = np.loadtxt(csv_file_path,
                                     dtype=np.float64,
                                     delimiter=',',
                                     skiprows=1)

    def combine_trajectory_csv(self, csv_dir_path, processed=True):
        """Combine the unprocessed trajectories in one .csv file

        Args:
            csv_dir_path: path of the .csv directory
        """
        if processed == True:
            files = os.path.join(csv_dir_path, "sim_data_proc_*.csv")
        else:
            files = os.path.join(csv_dir_path, "sim_data_*.csv")
        # list of merged files returned
        files = glob.glob(files)

        # joining files with concat and read_csv
        df = pd.concat(map(pd.read_csv, files), ignore_index=True)
        self.fieldnames = (df.columns).values.tolist()
        self.robot_info = df.to_numpy()

    def contact_info_processed_to_csv(self,
                                      csv_name,
                                      csv_dir_path,
                                      csv_name_norm,
                                      keep_traj=False):
        """After calling get_info_robot() and contact_info_processed_to_csv() method.
        Normalization of the dataset to obtain a distribution with mean = 0 and std = 1.
        Prints the normalized trajectories into .csv files in the folder csv_folder_resampled

        Args:
            n: number of trajectories
        """
        csv_full_path = os.path.join(csv_dir_path, csv_name)
        csv_full_path_norm = os.path.join(csv_dir_path, csv_name_norm)

        # Check if directory exists, otherwise create it
        if not os.path.isdir(csv_dir_path):
            os.makedirs(csv_dir_path)

        mean = self.robot_info.mean(0)
        std = self.robot_info.std(0)
        noise = np.random.normal(0, 0.01 / std[20], len(self.robot_info))

        self.robot_info_processed = np.concatenate(
            (
                (self.robot_info[:, :18] - mean[:18]) / std[:18],
                (((self.robot_info[:, 18:] - mean[18:]) / std[18:]).transpose() +
                 noise).transpose(),  #mean = 0, std = 1, Gaussian noise N(0, 0.05N)
            ),
            axis=1)

        with open(csv_full_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow(self.fieldnames)
            writer.writerows(np.array(self.robot_info_processed, dtype=np.float32))

        with open(csv_full_path_norm, "a") as g:
            writer = csv.writer(g)
            writer.writerow(self.fieldnames)
            writer.writerow(mean)
            writer.writerow(std)

    def plot_data_comparison(self):
        """Plot the info registered vs the info processed.
        After calling get_info_robot() and contact_info_processed_to_csv() method.
        
        Returns:
            fig
        """
        title = ['$x_{ee}$', '$\dot{x}_{ee}$', '$f_{z_{ee}}$']
        idx_vect = [2, 8, 20]
        udm = ['[m]', '[m/s]', '[N]']
        color = ["#37CAEC", "#2A93D5", "#125488"]

        fig, axs = plt.subplots(2, 3)
        step_vect = np.arange(0, len(self.robot_info), 1)

        for j in range(3):
            #first row
            axs[0, j].plot(step_vect,
                           self.robot_info_processed[:, idx_vect[j]],
                           color=color[j])
            axs[0, j].set_title(f"%s processed" % title[j])
            axs[0, j].ticklabel_format(style="sci", scilimits=(0, 0))
            axs[0, j].set_ylabel(udm[j])
            axs[0, j].grid()
            #second row
            axs[1, j].plot(step_vect, self.robot_info[:, idx_vect[j]], color=color[j])
            axs[1, j].set_title(title[j])
            axs[1, j].ticklabel_format(style="sci", scilimits=(0, 0))
            axs[1, j].set_ylabel(udm[j])
            axs[1, j].set_xlabel('steps')
            axs[1, j].grid()

        return [axs, fig]
