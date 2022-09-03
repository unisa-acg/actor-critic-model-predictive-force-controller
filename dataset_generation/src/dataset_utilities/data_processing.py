import csv
import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob


class DataProcessing:
    """
    Class that contains method to manipulate the data stored in the folder 'csv_folder' 
    (normalization and Gaussian noise addition) and store them in .csv files.
    This dataset will be used to train the neural network whose task is to learn the 
    forces produced by the robot-environment contact.
    """

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

    def contact_info_processed_to_csv(self, csv_name, csv_dir_path, csv_norm_data_name):
        """Normalization of the dataset to obtain a distribution with mean = 0 and 
        std = 1.
        Prints the normalized trajectories into .csv file in the folder processed_csv.
        After calling combine_trajectory_csv() method.

        Args:
            csv_name (string): name of the csv file
            csv_file_path (string): path of the csv file
        """

        csv_full_path = os.path.join(csv_dir_path, csv_name)
        csv_full_path_norm_data = os.path.join(csv_dir_path, csv_norm_data_name)

        # Check if directory exists, otherwise create it
        if not os.path.isdir(csv_dir_path):
            os.makedirs(csv_dir_path)

        mean = self.robot_info.mean(0)
        std = self.robot_info.std(0)
        noise = np.random.normal(0, 0.05 / 3 / std[20], len(self.robot_info))

        # Processing output: mean = 0, std = 1, Gaussian noise N(0, 0.05N)
        self.robot_info_processed = np.concatenate((
            (self.robot_info[:, :18] - mean[:18]) / std[:18],
            (((self.robot_info[:, 18:] - mean[18:]) / std[18:]).transpose() +
             noise).transpose(),
        ),
                                                   axis=1)

        with open(csv_full_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow(self.fieldnames)
            writer.writerows(np.array(self.robot_info_processed, dtype=np.float32))

        f.close()

        with open(csv_full_path_norm_data, "a") as g:
            writer = csv.writer(g)
            writer.writerow(self.fieldnames)
            writer.writerow(mean)
            writer.writerow(std)

        g.close()

    def plot_data_comparison(self):
        """Plot the information registered vs the information processed. Only the data 
        subsequently used are plotted and not all the data contained in the .csv file.
        After calling contact_info_processed_to_csv() method.
        
        Returns:
            fig
        """
        title = ['$x_{ee}$', '$\dot{x}_{ee}$', '$f_{z_{ee}}$']
        idx_vect = [2, 8, 20]
        udm = ['[m]', '[m/s]', '[N]']
        color = ["#37CAEC", "#2A93D5", "#125488"]

        _, axs = plt.subplots(2, 3)
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

        return
