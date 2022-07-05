import numpy as np
import csv
import os.path
from scipy.interpolate import interp1d


class TrajectoryResampler:
    """
    Class that contains method to resample and plot the trajectories from a trajectory matrix previously generated
    """

    def __init__(self):
        self.ncalls = 0

    def interp_traj(self, traj_matrix, time, new_time_step=0.002):
        """Resamples the trajectory `traj_matrix`, given a vector of sample times `time`, with a chosen time step `new_time_step` (default = 0.002). 

        Args:
            traj_matrix (NDArray): 2D trajectory matrix
            time (NDArray, optional): vector of times used in sampling the traj_matrix.
            new_time_step (float, optional): new time step for resampling. Defaults to 0.02.
            
        Returns:
            [resampled_df, data_frame]: returns the resampled data frame and the data frame constructed from the original trajectory `[Resampled df, Original df]`
        """
        new_time = np.arange(time[0], time[-1], new_time_step)

        resampled_df = np.stack(
            (
                interp1d(time, traj_matrix[:, 0], kind='cubic')(new_time),
                interp1d(time, traj_matrix[:, 1], kind='cubic')(new_time),
                interp1d(time, traj_matrix[:, 2], kind='linear')(new_time),
            ),
            axis=-1,
        )

        data_frame = traj_matrix
        self.traj_resampled = resampled_df

        return [resampled_df, data_frame]

    def traj_res_csv(self, csv_file_path):
        """Prints resampled trajectory into a .csv file in the csv_file_path location

        Args:
            csv_file_path: path of the .csv file where the data will be saved
        """
        with open(csv_file_path, "a") as f:
            writer = csv.writer(f)
            header = ['x_res', 'y_res', 'f_res']
            writer.writerow(header)
            writer.writerows(self.traj_resampled)

    def read_traj_from_csv(self, csv_file_path):
        """Reads and return the data from .csv files

        Args:
            csv_file_path: path of the .csv file whose data will be read

        Returns:
            df: NDArray containing the data read from the csv
        """

        df = np.loadtxt(csv_file_path,
                        dtype=np.float32,
                        delimiter=',',
                        skiprows=1)

        return df