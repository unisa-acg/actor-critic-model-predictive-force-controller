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

    def combine_trajectory_csv(self, csv_folder, processed=True):
        if processed == True:
            csv_files_path = csv_folder + '/processed_csv'
            # files = os.path.join(csv_files_path, "sim_data_proc_*.csv")
            files = os.path.join(csv_files_path, "sim_data_proc_manipolato_2.csv")
        else:
            csv_files_path = csv_folder + '/unprocessed_csv'
            files = os.path.join(csv_files_path, "sim_data_*.csv")
        # list of merged files returned
        files = glob.glob(files)

        # joining files with concat and read_csv
        df = pd.concat(map(pd.read_csv, files), ignore_index=True)
        self.fieldnames = (df.columns).values.tolist()
        self.robot_info = df.to_numpy()

    def contact_info_processed_to_csv(self, csv_file_path, keep_traj=False):
        """After calling get_info_robot() and contact_info_processed_to_csv() method.
        Normalization of the dataset to obtain a distribution with mean = 0 and std = 1.
        Prints the normalized trajectories into .csv files in the folder csv_folder_resampled

        Args:
            n: number of trajectories
        """
        if keep_traj == False:
            keep = "n"

        mean = self.robot_info.mean(0)
        std = self.robot_info.std(0)

        noise = np.random.normal(0, 0.01 / std[20], len(self.robot_info))
        # noise = np.random.normal(0, 0.05 / 3 / 15.35, len(self.robot_info))

        # print(noise)
        # print(std[20])
        # print(mean[20])
        # self.robot_info[:,20] = self.robot_info[:,20] + noise
        # self.robot_info_processed = self.robot_info

        self.robot_info_processed = np.concatenate(
            (
                (self.robot_info[:, :18] - mean[:18]) / std[:18],
                (((self.robot_info[:, 18:] - mean[18:]) / std[18:]).transpose() +
                 noise).transpose(),  #mean = 0, std = 1, Gaussian noise N(0, 0.005)
            ),
            axis=1)

        #self.robot_info_processed = (((self.robot_info - mean)/std).transpose() + noise).transpose()   #mean = 0, std = 1, Gaussian noise N(0, 0.005)

        # Check if file .csv exists or not, ask the user to keep it and append to it or replace it
        file_exists = os.path.exists(csv_file_path)
        if (file_exists == False):
            print("---")
            print(
                "The 📄.csv file for the trajectories data doesn't exists, creating it..."
            )
            open(csv_file_path, 'a')
            print("---")
            #self.ncalls = 1

        if (file_exists == True):
            print("---")
            if keep_traj != False:
                keep = input(
                    "The 📄.csv file for the trajectories data already exists, do you want to keep it? [y/n] "
                )
            if keep == "n":
                print("File 📄.csv replaced")
                os.remove(csv_file_path)
                open(csv_file_path, 'a')
            print("---")

        f = open(csv_file_path, 'a')
        writer = csv.writer(f)
        writer.writerow(self.fieldnames)
        writer.writerows(np.array(self.robot_info_processed, dtype=np.float32))

        f.close()

        g = open('csv_folder/processed_csv/norm_data.csv', 'a')
        writer = csv.writer(g)
        writer.writerow(self.fieldnames)
        writer.writerow(mean)
        writer.writerow(std)

        g.close()

    def plot_data_comparison(self):
        """Plot the info registered vs the info processed.
        After calling get_info_robot() and contact_info_processed_to_csv() method.
        
        Returns:
            fig
        """
        title = [
            '$pos_x$',
            '$pos_y$',
            '$pos_z$',
            '$ori_x$',
            '$ori_y$',
            '$ori_z$',
            '$vel_x$',
            '$vel_y$',
            '$vel_z$',
            '$omg_x$',
            '$omg_y$',
            '$omg_z$',
            '$\Delta pos_x$',
            '$\Delta pos_y$',
            '$\Delta pos_z$',
            '$\Delta ori_x$',
            '$\Delta ori_y$',
            '$\Delta ori_z$',
            #'$f_x$','$f_y$','$f_z$',
            '$f_{x+1}$',
            '$f_{y+1}$',
            '$f_{z+1}$',
            '$torque_{x+1}$',
            '$torque_{y+1}$',
            '$torque_{z+1}$'
        ]
        #'$torque_x$','$torque_y$','$torque_z$'
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
