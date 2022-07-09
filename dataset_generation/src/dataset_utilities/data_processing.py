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

    def contact_info_processed_to_csv(self, csv_name, csv_dir_path):
        """Normalization of the dataset to obtain a distribution with mean = 0 and 
        std = 1.
        Prints the normalized trajectories into .csv file in the folder processed_csv.
        After calling combine_trajectory_csv() method.

        Args:
            csv_name (string): name of the csv file
            csv_file_path (string): path of the csv file
        """

        csv_full_path = os.path.join(csv_dir_path, csv_name)

        # Check if directory exists, otherwise create it
        if not os.path.isdir(csv_dir_path):
            os.makedirs(csv_dir_path)

        mean = self.robot_info.mean(0)
        std = self.robot_info.std(0)
        noise = np.random.normal(0, 0.005, len(self.robot_info))

        #self.robot_info_processed = (self.robot_info - mean)/std

        self.robot_info_processed = (
            ((self.robot_info - mean) / std).transpose() +
            noise).transpose()  #mean = 0, std = 1, Gaussian noise N(0, 0.005)

        with open(csv_full_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow(self.fieldnames)
            writer.writerows(np.array(self.robot_info_processed, dtype=np.float32))

    def plot_data_comparison(self):
        """Plot the information registered vs the information processed.
        After calling contact_info_processed_to_csv() method.
        
        Returns:
            fig
        """
        title = [
            '$pos_x$', '$pos_y$', '$pos_z$', '$ori_x$', '$ori_y$', '$ori_z$', '$vel_x$',
            '$vel_y$', '$vel_z$', '$omg_x$', '$omg_y$', '$omg_z$', '$\Delta pos_x$',
            '$\Delta pos_y$', '$\Delta pos_z$', '$f_x$', '$f_y$', '$f_z$', '$torque_x$',
            '$torque_y$', '$torque_z$', '$f_{x+1}$', '$f_{y+1}$', '$f_{z+1}$',
            '$torque_{x+1}$', '$torque_{y+1}$', '$torque_{z+1}$'
        ]

        n_graph = 0
        for i in range(4):
            fig, axs = plt.subplots(2, 6)
            step_vect = np.arange(0, len(self.robot_info), 1)
            for j in range(6):
                axs[0, j].plot(step_vect, self.robot_info_processed[:, n_graph])
                axs[0, j].set_title(f"%s processed" % title[n_graph])
                axs[0, j].grid()
                axs[1, j].plot(step_vect, self.robot_info[:, n_graph])
                axs[1, j].set_title(title[n_graph])
                axs[1, j].grid()
                n_graph += 1

            for ax in axs.flat:
                ax.set(xlabel='Steps', ylabel='$[N]$')

        return [axs, fig]
