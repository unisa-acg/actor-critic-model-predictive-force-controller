from traja import resample_time
import traja
import matplotlib.pyplot as plt
from matplotlib.pyplot import ion, show
import numpy as np
import csv
import decimal
import os.path
import pandas as pd
from scipy.interpolate import interp1d
from scipy import interpolate

from itertools import cycle
from mpl_toolkits.mplot3d import Axes3D


class traj_utilities:
    """
    Class that contains method to resample and plot the trajectories from a trajectory matrix
    """

    def __init__(self):
        self.ncalls = 0

    def convert_to_traj_format(self, new_time_step):
        """Auxiliary function to convert the time-step (ms) value in a Traja format

        Args:
            new_time_step (float): time step to be converted

        Returns:
            float: time step in Traja format
        """
        decimal_values = decimal.Decimal(
            str(new_time_step)).as_tuple().exponent
        last_digit = int(repr(new_time_step)[-1])
        exponent = 3 - abs(decimal_values)
        ms = str(10**exponent * last_digit) + "L"

        return ms

    def traj_to_df(self, traj_matrix, time):
        """Creates the data frame from the trajectory matrix (2D or 3D)

        Args:
            traj_matrix (NDArray): matrix with column vectors the points (x,y) or (x,y,z) of the trajectory
            time (NDArray): time vector

        Returns:
            TrajaDataFrame: trajectory data frame
        """
        # Check if trajectory is on 3D or 2D and construct dataframe from trajectory
        if traj_matrix.shape[1] == 2:  # 2D traj

            data_frame = traja.TrajaDataFrame({
                "x":
                traj_matrix[:, 0].transpose(),
                "y":
                traj_matrix[:, 1].transpose(),
                "time":
                time.transpose(),
            })

        elif traj_matrix.shape[1] == 3:  # 3D traj

            data_frame = traja.TrajaDataFrame({
                "x":
                traj_matrix[:, 0].transpose(),
                "y":
                traj_matrix[:, 1].transpose(),
                "z":
                traj_matrix[:, 2].transpose(),
                "time":
                time.transpose(),
            })
        return data_frame

    def interp_traj(self,
                    traj_matrix,
                    time,
                    old_time_step=None,
                    new_time_step=0.002,
                    time_vect=None,
                    lin_interp=True):
        """Resample the trajectory, originally sampled with `old_time_step` or given a vector of sample times `time_vect`, with a chosen time step `new_time_step` (default = 0.02). Either a new time step or a new time vector must be provided

        Args:
            traj_matrix (NDArray): 2D or 3D trajecctory matrix
            old_time_step (float, optional): time step used in sampling the traj_matrix. Defaults to None.
            new_time_step (float, optional): new time step for resampling. Defaults to 0.02.
            time_vect (NDArray, optional): vector of times used in sampling the traj_matrix. Defaults to None.

        Returns:
            [TrajaDataFrame,TrajaDataFrame]: returns the resampled data frame and the data frame constructed from the original trajectory `[Resampled df, Original df]`
        """
        if lin_interp == True:
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
                axis=-1,
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

        else:
            if time_vect is None and old_time_step is None:
                raise "Either time_vect OR old_time_step parameter must be provided"

            if time_vect != None:
                try:
                    traj_matrix.shape[0] == len(time_vect)
                except:
                    print(
                        "The time vector doesn't have the same dimension of the trajectory matrix"
                    )

                old_time_vect = time_vect
            else:
                old_time_vect = np.arange(traj_matrix.shape[0]) * old_time_step

            # Generate the data frame to be resampled
            data_frame = self.traj_to_df(traj_matrix, time)
            self.old_time_vect = old_time_vect
            self.new_time_vect = np.arange(old_time_vect[0],
                                           old_time_vect[-1] + new_time_step,
                                           new_time_step)
            # Resample the original dataframe with the same step in order to be in the same format of the other
            ms_old = self.convert_to_traj_format(old_time_step)
            data_frame = resample_time(data_frame, ms_old)
            # Resample the dataframe
            ms = self.convert_to_traj_format(new_time_step)
            resampled_df = resample_time(data_frame, ms)
            resampled_df.head()

        return [resampled_df, data_frame]

    def plot_traj_2d(self, data_frame):
        """Auxiliary function for plotting the trajectory in 2D case

        Args:
            data_frame (TrajaDataFrame): Data frame to be plotted

        Returns:
            fig: plot fo the trajectory in 2D
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)
        time_vect = np.zeros(len(data_frame.index))
        time_vect[0] = data_frame.index[0].to_numpy()
        for i in range(1, len(data_frame.index)):
            time_vect[i] = (time_vect[i - 1] +
                            (data_frame.index[i] -
                             data_frame.index[i - 1]).total_seconds())
        ax.plot(data_frame["x"].to_numpy(), data_frame["y"].to_numpy())
        p = ax.scatter(
            data_frame["x"].to_numpy(),
            data_frame["y"].to_numpy(),
            c=time_vect,
            cmap="winter_r",
        )
        colorbar = fig.colorbar(p)
        colorbar.ax.get_yaxis().labelpad = 15
        colorbar.set_label("Time", rotation=270)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("2D trajectory in time")

        return fig

    def plot_traj_3d(self, data_frame, lin_interp=True):
        """Auxiliary function for plotting the trajectory in 3D case

        Args:
            data_frame (TrajaDataFrame): Data frame to be plotted

        Returns:
            fig: plot fo the trajectory in 3D
        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        if lin_interp == True:
            ax.plot(data_frame[:, 0], data_frame[:, 1], data_frame[:, 2])

        else:
            time_vect = np.zeros(len(data_frame.index))
            time_vect[0] = data_frame.index[0].to_numpy()
            for i in range(1, len(data_frame.index)):
                time_vect[i] = (time_vect[i - 1] +
                                (data_frame.index[i] -
                                 data_frame.index[i - 1]).total_seconds())

            ax.plot(
                data_frame["x"].to_numpy(),
                data_frame["y"].to_numpy(),
                data_frame["z"].to_numpy(),
            )
            p = ax.scatter3D(
                data_frame["x"].to_numpy(),
                data_frame["y"].to_numpy(),
                data_frame["z"].to_numpy(),
                c=time_vect,
                cmap="winter_r",
            )

            colorbar = fig.colorbar(p)
            colorbar.ax.get_yaxis().labelpad = 15
            colorbar.set_label("Time", rotation=270)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("3D trajectory in time")

        return fig

    def plot_traj_from_df(self, data_frame):
        """Plots the trajectory (2D or 3D) stored in a Traja dataframe

        Args:
            data_frame (TrajaDataFrame): data frame in which the trajectory (2D or 3D) is stored
        """
        # Instantiate figure with dimensionality 2D or 3D
        header = data_frame.columns
        if "z" not in header:
            fig = self.plot_traj_2d(data_frame)
        elif "z" in header:
            fig = self.plot_traj_3d(data_frame)

    # ---------------------------------------------------------------------------- #
    #                       Save trajectories informations to csv                  #
    # ---------------------------------------------------------------------------- #

    def traj_res_csv(self,
                     num_traj,
                     traj_matrix,
                     time,
                     old_time_step,
                     new_time_step=0.002,
                     keep_traj=False,
                     lin_interp=True):
        """Prints all the random trajectories into a .csv file in the current location

        Args:
            length_time_vect: length of the vector time of the starting trajectory (not already resampled)
        """
        if keep_traj == False:
            keep = "n"

        # Check if file .csv exists or not, ask the user to keep it and append to it or replace it
        file_exists = os.path.exists("traj_resampled.csv")
        if (file_exists == False) and (self.ncalls == 0):
            print("---")
            print(
                "The 📄.csv file for the trajectories data doesn't exists, creating it..."
            )
            open("traj_resampled.csv", "x")
            print("---")

        if (file_exists == True) and (self.ncalls == 0):
            print("---")
            if keep_traj != False:
                keep = input(
                    "The 📄.csv file for the trajectories data already exists, do you want to keep it? [y/n] "
                )
            if keep == "n":
                print("File 📄.csv replaced")
                os.remove("traj_resampled.csv")
                open("traj_resampled.csv", "x")
            print("---")

        # Trajectories are appended with the keyword "trajectory"

        trajectory_string = ['trajectory']

        f = open('traj_resampled.csv', 'a')
        writer = csv.writer(f)

        for i in range(num_traj):
            [traj_resampled,
             data_frame] = self.interp_traj(traj_matrix, time, old_time_step,
                                            new_time_step)
            writer.writerow(trajectory_string)
            if lin_interp == True:
                writer.writerows(traj_resampled)
            else:
                writer.writerows(traj_resampled.values)

        f.close()

    def read_traj_from_csv(self, csv_file_path):

        with open(csv_file_path) as csv_file:

            content = csv_file.readlines()
            content = [x[:-1] for x in content]

            index_vect = np.where(np.isin(content, "trajectory"))
            index_vect = np.array(index_vect)
            index_vect = np.append(index_vect, len(content))
            traj_count = len(index_vect) - 1
            for i in range(len(index_vect) - 1):

                traj_df = content[index_vect[i] + 1:index_vect[i + 1]]
                traj_df = np.array([x.split(",") for x in traj_df])
                if i == 0:
                    df = pd.DataFrame(
                        traj_df,
                        columns=["x" + str(i), "y" + str(i), "f" + str(i)])
                else:
                    df["x" + str(i)] = traj_df[:, 0]
                    df["y" + str(i)] = traj_df[:, 1]
                    df["f" + str(i)] = traj_df[:, 2]

            print(f"Found {traj_count} trajectories in the csv file")

            return [df, traj_count]