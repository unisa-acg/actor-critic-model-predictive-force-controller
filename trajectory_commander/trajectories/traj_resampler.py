import numpy as np
import csv
import os.path
from scipy.interpolate import interp1d

class TrajectoryResampler:
    """
    Class that contains method to resample and plot the trajectories from a trajectory matrix
    """
    def __init__(self):
        self.ncalls = 0

    def interp_traj(self, traj_matrix, time, new_time_step=0.002):
        """Resample the trajectory `traj_matrix`, given a vector of sample times `time`, with a chosen time step `new_time_step` (default = 0.02). 

        Args:
            traj_matrix (NDArray): 2D trajectory matrix
            time (NDArray, optional): vector of times used in sampling the traj_matrix.
            new_time_step (float, optional): new time step for resampling. Defaults to 0.02.
            
        Returns:
            [resampled_df, data_frame]: returns the resampled data frame and the data frame constructed from the original trajectory `[Resampled df, Original df]`
        """
        new_time = np.arange(time[0], time[-1], new_time_step)
        # resampled_df = np.stack(
        #     (
        #     np.interp(new_time, time, traj_matrix[:, 0]),
        #     np.interp(new_time, time, traj_matrix[:, 1]),
        #     np.interp(new_time, time, traj_matrix[:, 2]),
        #     ),
        #     axis = -1,
        #     )
        resampled_df = np.stack(
            (
            interp1d(time, traj_matrix[:, 0], kind='cubic')(new_time),
            interp1d(time, traj_matrix[:, 1], kind='cubic')(new_time),
            interp1d(time, traj_matrix[:, 2], kind='linear')(new_time),
            ),
            axis = -1,
            )
        # resampled_df = np.stack(
        #     (
        #     interpolate.splev(new_time, interpolate.splrep(time, traj_matrix[:, 0], s=0), der=0),
        #     interpolate.splev(new_time, interpolate.splrep(time, traj_matrix[:, 1], s=0), der=0),
        #     interpolate.splev(new_time, interpolate.splrep(time, traj_matrix[:, 2], s=0), der=0),
        #     ),
        #     axis = -1,
        #     )
        data_frame = traj_matrix

        self.traj_resampled = resampled_df
        return [resampled_df, data_frame]

    def traj_res_csv(self, csv_file_path, keep_traj = False):
        """Prints resampled trajectory into a .csv file in the csv_file_path location

        Args:
            csv_file_path: path of the .csv file where the data will be saved
        """
        if keep_traj == False:
            keep = "n"
        
        # Check if file .csv exists or not, ask the user to keep it and append to it or replace it
        file_exists = os.path.exists(csv_file_path)
        if (file_exists == False) and (self.ncalls == 0):
            print("---")
            print("The 📄.csv file for the trajectories data doesn't exists, creating it...")
            open(csv_file_path, 'a')
            print("---")
            #self.ncalls = 1

        if (file_exists == True) and (self.ncalls == 0):
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
            #self.ncalls = 1

        f = open(csv_file_path, 'a')
        writer = csv.writer(f)

        header = ['x_res', 'y_res', 'f_res']
        writer.writerow(header)
        writer.writerows(self.traj_resampled)

        f.close()

    def read_traj_from_csv(self, csv_file_path):
        """It reads the data from .csv files

        Args:
            csv_file_path: path of the .csv file whose data will be red
        """
        df = np.loadtxt(csv_file_path, dtype=np.float32, delimiter=',', skiprows=1)

        return df