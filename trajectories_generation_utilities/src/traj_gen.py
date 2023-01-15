from genericpath import exists
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import os.path

from torch import square


class TrajectoryGenerator():
    """
    Class that contains method to generate, plot and save on a .csv file a trajectory 
    given the parameters described in ``traj_gen()`` and a trend of reference forces. 
    The data are stored in a folder selected by the user.
    """

    def __init__(self, timestep=0.01):
        """
        Assign default timestep
        """
        self.ts = timestep

    # ---------------------------------------------------------------------------- #
    # Trajectory generation section

    def traj_gen(self, waypoints: list, traj_types: list, traj_parameters: list,
                 traj_timestamps: list, force_ref_types: list, force_ref_params: list):
        """Generate a trajectory made by sub-trajectory of different types.

        Supported parameters:
            - Trajectory types: 
                `line`,`circle`,`sine_curve`

            - Trajectory parameters:
                - `line`: None
                - `circle`: [radius]
                - `sine_curve`: [amplitude,frequency]

            - Force reference types: 
                `cnst`,`ramp`, `sine_curve`

            - Force reference parameters:
                - `cnst`: value
                - `ramp`: [initial value,final value] 
                - `sine_curve`: [amplitude,frequency]

        Returns:
            [x,y,f]: 
        """
        x_traj = []
        y_traj = []
        f_ref_traj = []

        if len(traj_types) != len(traj_parameters):
            raise Exception("Trajectories types and parameters lengths do not match")

        for i in range(len(traj_types)):
            xi = waypoints[i]
            xf = waypoints[i + 1]

            duration = traj_timestamps[i + 1] - traj_timestamps[i]

            if traj_types[i] == 'line':
                [x, y] = self._line(xi, xf, self.ts, duration)
            elif traj_types[i] == 'circle':
                [x, y] = self._circle(xi, xf, self.ts, duration, traj_parameters[i])
            elif traj_types[i] == 'sine_curve':
                [x, y] = self._sine_curve(xi, xf, self.ts, duration,
                                          traj_parameters[i][0], traj_parameters[i][1])
            else:
                raise Exception("No matching trajectory type found for trajectory" +
                                str(i + 1))

            x_traj = np.append(x_traj, x)
            y_traj = np.append(y_traj, y)

            if force_ref_types[i] == 'cnst':
                [s, f_ref] = self._line((0, force_ref_params[i]),
                                        (1, force_ref_params[i]), self.ts, duration)
            elif force_ref_types[i] == 'ramp':
                [s, f_ref] = self._line((0, force_ref_params[i][0]),
                                        (1, force_ref_params[i][1]), self.ts, duration)
            elif force_ref_types[i] == 'sine_curve':
                [s, f_ref] = self._sine_curve_f(
                    (0, force_ref_params[i][0]), (1, force_ref_params[i][1]), self.ts,
                    duration, force_ref_params[i][2], force_ref_params[i][3])
            else:
                raise Exception("No matching trajectory type found for trajectory" +
                                str(i + 1))

            f_ref_traj = np.append(f_ref_traj, f_ref)

        self.x_traj_tot = x_traj
        self.y_traj_tot = y_traj
        self.f_ref_tot = f_ref_traj

        return [x_traj, y_traj, f_ref_traj]

    def _line(self, xi, xf, ts, t_tot):
        """Generate a circle sub-trajectory

        Args:
            xi (tuple): initial point of the line
            xf (tuple): final point of the line
            ts (float): timestep for the trajectory generation
            t_tot (float): total duration of the trajectory

        Returns:
            [x,y]: vectors containing x and y points of the trajectory
        """
        if (xi == xf):
            raise Exception('Final point and initial point must be different')

        n_points = round(t_tot / ts)

        def line_eq(x):
            return m * x + q

        if (xi[0] - xf[0]) == 0:
            x = np.linspace(xi[0], xf[0], n_points)
            y = np.linspace(xi[1], xf[1], n_points)
        else:
            m = (xi[1] - xf[1]) / (xi[0] - xf[0])
            q = (xi[0] * xf[1] - xf[0] * xi[1]) / (xi[0] - xf[0])
            x = np.linspace(xi[0], xf[0], n_points)
            y = line_eq(x)

        return [x, y]

    def _circle(self, xi, xf, ts, t_tot, r):
        """Generate a circle sub-trajectory

        Args:
            xi (tuple): initial point of the circle
            xf (tuple): final point of the circle
            ts (float): timestep for the trajectory generation
            t_tot (float): total duration of the trajectory
            r (float): radius of the circle

        Returns:
            [x,y]: vectors containing x and y points of the trajectory
        """
        if (xi[0] != xf[0]) and (xi[1] != xf[1]):
            raise Exception('Final point and initial point must be equal')

        n_points = round(t_tot / ts)

        def x_c(alpha):
            return r * np.cos(alpha * 2 * np.pi / 360)

        def y_c(alpha):
            return r * np.sin(alpha * 2 * np.pi / 360)

        alpha_vect = np.linspace(0, 360, n_points)
        x = (x_c(alpha_vect)).transpose()
        y = (y_c(alpha_vect)).transpose()

        x = x + xi[0] - r
        y = y + xi[1]

        return [x, y]

    def _sine_curve(self, xi, xf, ts, t_tot, ampl, freq):
        """Generate a sine sub-trajectory

        Args:
            xi (tuple): initial point of the sine wave
            xf (tuple): final point of the sine wave
            ts (float): timestep for the trajectory generation
            t_tot (float): total duration of the trajectory
            ampl (float): amplitude of the sine wave
            freq (int): frequency of the sine wave

        Returns:
            [x,y]: vectors containing x and y points of the trajectory
        """
        if (xi[0] == xf[0]) and (xi[1] == xf[1]):
            raise Exception('Final point and initial point must be different')
        if (isinstance(freq, int)) != True:
            raise Exception('Frequency must be an integer number')

        def sin_eq(x):
            return ampl * np.sin(freq * x)

        n_points = round(t_tot / ts)

        x = np.linspace(0, np.sqrt((xf[0] - xi[0])**2 + (xf[1] - xi[1])**2), n_points)
        y = sin_eq(np.linspace(0, 1, n_points) * 2 * np.pi)

        # Sine rotation
        m = (xf[1] - xi[1]) / (xf[0] - xi[0])

        if ((xf[0] - xi[0] < 0)):
            theta = np.arctan(m) + np.pi
        else:
            theta = np.arctan(m)

        rot_mtr = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
        [x_rot, y_rot] = rot_mtr.dot(np.array([x, y]))
        x_rot = x_rot + xi[0]
        y_rot = y_rot + xi[1]

        return [x_rot, y_rot]

    def _sine_curve_f(self, xi, xf, ts, t_tot, ampl, freq):
        """Generate a sine sub-trajectory

        Args:
            xi (tuple): initial point of the sine wave
            xf (tuple): final point of the sine wave
            ts (float): timestep for the trajectory generation
            t_tot (float): total duration of the trajectory
            ampl (float): amplitude of the sine wave
            freq (int): frequency of the sine wave

        Returns:
            [x,y]: vectors containing x and y points of the trajectory
        """
        if (xi[0] == xf[0]) and (xi[1] == xf[1]):
            raise Exception('Final point and initial point must be different')
        if (isinstance(freq, int)) != True:
            raise Exception('Frequency must be an integer number')

        def sin_eq(x):
            return ampl * np.sin(freq * x)

        n_points = round(t_tot / ts)

        x = np.linspace(0, np.sqrt((xf[1] - xi[1])**2 + (xf[1] - xi[1])**2), n_points)
        y = sin_eq(np.linspace(0, 1, n_points) * 2 * np.pi)

        # Sine rotation
        m = (xf[1] - xi[1]) / (xf[1] - xi[1])

        if ((xf[1] - xi[1] < 0)):
            theta = np.arctan(m) + np.pi
        else:
            theta = np.arctan(m)

        rot_mtr = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
        [x_rot, y_rot] = rot_mtr.dot(np.array([x, y]))
        x_rot = x_rot + xi[0]
        y_rot = y_rot + xi[1]

        return [x_rot, y_rot]

    # ---------------------------------------------------------------------------- #
    # Plot section

    def plot_traj(self, x, y, params_randomizer):
        """After calling ``traj_gen()`` method, it plots the points of the 
        generated trajectory in the operative space

        Args:
            x (NDArray): vector of x positions 
            y (NDArray): vector of y positions 
            traj_timestamps (dict): dictionary containing the parameter for 
            trajectory generation
        
        Returns:
            Figure: xy-plane trajectory plot
        """
        fig = plt.figure()
        plt.scatter(x,
                    y,
                    color='#005B96',
                    marker='o',
                    linewidth=2,
                    label='Generated',
                    zorder=2)

        plt.scatter(x[0],
                    y[0],
                    color='red',
                    marker='o',
                    linewidth=2,
                    label='Starting point',
                    zorder=2)

        xy = (params_randomizer['operating_zone_points'][0][1],
              params_randomizer['operating_zone_points'][0][0])

        width = abs(params_randomizer['operating_zone_points'][0][1]) + abs(
            params_randomizer['operating_zone_points'][1][1])

        height = abs(params_randomizer['operating_zone_points'][0][0]) + abs(
            params_randomizer['operating_zone_points'][1][0])

        patches.Rectangle(xy, width, height)

        rect = patches.Rectangle(xy,
                                 width,
                                 height,
                                 fill=False,
                                 color='#03396c',
                                 linewidth=2)
        #facecolor="red")
        plt.gca().add_patch(rect)

        # plt.plot(params_randomizer['operating_zone_points'][0][1],
        #          params_randomizer['operating_zone_points'][0][0],
        #          '*',
        #          markersize=15)
        # plt.plot(params_randomizer['operating_zone_points'][0][1],
        #          -params_randomizer['operating_zone_points'][0][0],
        #          '*',
        #          markersize=15)
        # plt.plot(params_randomizer['operating_zone_points'][1][1],
        #          params_randomizer['operating_zone_points'][1][0],
        #          '*',
        #          markersize=15)
        # plt.plot(params_randomizer['operating_zone_points'][1][1],
        #          -params_randomizer['operating_zone_points'][1][0],
        #          '*',
        #          markersize=15)
        plt.xlabel('$x-axis$')
        plt.ylabel('$y-axis$')
        # plt.title('$Position \; reference$')
        plt.grid()
        return fig

    def plot_force_ref(self, f_ref, traj_timestamps):
        """After calling `traj_gen()` method, it plots the points of the 
        generated force reference

        Args:
            f_ref (NDArray): vector of reference forces 
            traj_timestamps (NDArray): vector of timestamps

        Returns:
            Figure: force reference plot
        """
        t_dis = np.arange(traj_timestamps[0], traj_timestamps[-1], self.ts)
        fig = plt.figure()
        plt.scatter(t_dis,
                    f_ref,
                    color='#005B96',
                    marker='o',
                    linewidth=2,
                    label='Generated',
                    zorder=2)
        plt.xlabel('$Time \, [s]$')
        plt.ylabel('$Force \, [N]$')
        # plt.title('$Force \; reference$')
        plt.grid()
        return fig

    # ---------------------------------------------------------------------------- #
    # CSV section

    def print_to_csv(
        self,
        csv_name,
        csv_dir_path,
    ):
        """Prints the trajectory generated into a csv file in the `csv_file_path` location 

        Args:
            csv_name (string): name of the csv file
            csv_file_path (string): path of the csv file
        """

        csv_full_path = os.path.join(csv_dir_path, csv_name)

        # Check if directory exists, otherwise create it
        if not os.path.isdir(csv_dir_path):
            os.makedirs(csv_dir_path)

        if os.path.exists(csv_full_path):
            os.remove(csv_full_path)

        # Write the trajectory to the csv
        with open(csv_full_path, "a") as f:
            writer = csv.writer(f)
            header = ['x', 'y', 'f']
            trajectory = np.stack((self.x_traj_tot, self.y_traj_tot, self.f_ref_tot),
                                  axis=-1)
            writer.writerow(header)
            writer.writerows(trajectory)