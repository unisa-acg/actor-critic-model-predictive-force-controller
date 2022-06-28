import numpy as np
import matplotlib.pyplot as plt
import csv
import os.path


class TrajectoryGenerator():
    """
    Class that contains method to generate, plot and save on a .csv file a trajectory given the parameters described in ``traj_gen()`` and and a trend of reference forces. 
    The data are stored in a folder selected by the user.

    Methods
    -------
    * __init__(): instantiate the variables need for the other methods

    * traj_gen(self, waypoints: list, traj_types: list,
                 traj_parameters: list, traj_timestamps: list,
                 force_ref_types: list, force_ref_params: list):
        It generates the trajectory consistently with the input lists.
        It exploits the subfunction ``_line()``, ``_circle()`` and ``_sine_curve()``

    * plot_traj(param_randomizer): after calling ``traj_gen()`` method, it plots the points of the generated trajectory and the operative space

    * plot_traj(param_randomizer): after calling ``traj_gen()`` method, it plots the points of the generated force reference

    * print_to_csv(csv_file_path): after calling ``traj_gen()`` method, stores all the generated trajectory into a .csv file in the specified folder
    """

    def __init__(self):
        self.ts = 0.1
        self.ncalls = 0

    def traj_gen(self, waypoints: list, traj_types: list,
                 traj_parameters: list, traj_timestamps: list,
                 force_ref_types: list, force_ref_params: list):
        """
            Supported types:
            - Trajectory types: 
                'line','circle','sine_curve'

            - Trajectory parameters:
                - 'line': None
                - 'circle': [radius]
                - 'sine_curve': [amplitude,frequency]

            - Force reference types: 
                'cnst','ramp'

            - Force reference parameters:
                - 'cnst': value
                - 'ramp': [initial value,final value] 
        """

        x_traj = []
        y_traj = []
        f_ref_traj = []

        if len(traj_types) != len(traj_parameters):
            raise Exception(
                "Trajectories types and parameters lengths do not match")

        for i in range(len(traj_types)):
            xi = waypoints[i]
            xf = waypoints[i + 1]

            duration = traj_timestamps[i + 1] - traj_timestamps[i]

            if traj_types[i] == 'line':
                [x, y] = self._line(xi, xf, self.ts, duration)
            elif traj_types[i] == 'circle':
                [x, y] = self._circle(xi, xf, self.ts, duration,
                                      traj_parameters[i])
            elif traj_types[i] == 'sine_curve':
                [x, y] = self._sine_curve(xi, xf, self.ts, duration,
                                          traj_parameters[i][0],
                                          traj_parameters[i][1])
            else:
                raise Exception(
                    "No matching trajectory type found for trajectory" +
                    str(i + 1))

            x_traj = np.append(x_traj, x)
            y_traj = np.append(y_traj, y)

            if force_ref_types[i] == 'cnst':
                [s, f_ref] = self._line((0, force_ref_params[i]),
                                        (1, force_ref_params[i]), self.ts,
                                        duration)
            elif force_ref_types[i] == 'ramp':
                [s, f_ref] = self._line((0, force_ref_params[i][0]),
                                        (1, force_ref_params[i][1]), self.ts,
                                        duration)
            else:
                raise Exception(
                    "No matching trajectory type found for trajectory" +
                    str(i + 1))

            f_ref_traj = np.append(f_ref_traj, f_ref)

        self.x_traj_tot = x_traj
        self.y_traj_tot = y_traj
        self.f_ref_tot = f_ref_traj

        return [x_traj, y_traj, f_ref_traj]

    def _line(self, xi, xf, ts, t_tot):
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
        if (xi[0] == xf[0]) and (xi[1] == xf[1]):
            raise Exception('Final point and initial point must be different')
        if (isinstance(freq, int)) != True:
            raise Exception('Frequency must be an integer number')

        def sin_eq(x):
            return ampl * np.sin(freq * x)

        n_points = round(t_tot / ts)

        x = np.linspace(0, 1, n_points)

        y = sin_eq(x * 2 * np.pi)

        x = np.linspace(0, np.sqrt((xf[0] - xi[0])**2 + (xf[1] - xi[1])**2),
                        n_points)
        
        # sine rotation
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

    # ---------------------------------------------------------------------------- #

    def plot_traj(self, x, y, params_randomizer):
        """After calling ``traj_gen()`` method, it plots the points of the generated trajectory and the operative space
        
        Returns:
            fig
        """
        fig = plt.figure()
        plt.plot(x, y, 'ro', linewidth=2, label='Generated')
        plt.plot(params_randomizer['operating_zone_points'][0][1],
            params_randomizer['operating_zone_points'][0][0],
            '*',
            markersize=15)
        plt.plot(params_randomizer['operating_zone_points'][0][1],
            -params_randomizer['operating_zone_points'][0][0],
            '*',
            markersize=15)
        plt.plot(params_randomizer['operating_zone_points'][1][1],
            params_randomizer['operating_zone_points'][1][0],
            '*',
            markersize=15)
        plt.plot(params_randomizer['operating_zone_points'][1][1],
            -params_randomizer['operating_zone_points'][1][0],
            '*',
            markersize=15)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('$Position \; reference$')
        plt.grid()
        return fig

    def plot_force_ref(self, f_ref, traj_timestamps):
        """After calling ``traj_gen()`` method, it plots the points of the generated force reference
        
        Returns:
            fig
        """
        t_dis = np.arange(traj_timestamps[0], traj_timestamps[-1],
                          self.ts)
        fig = plt.figure()
        plt.plot(t_dis, f_ref, 'ro', linewidth=2, label='Generated')
        plt.xlabel('$Time$')
        plt.ylabel('$[N]$')
        plt.title('$Force \; reference$')
        plt.grid()
        return fig

    # ---------------------------------------------------------------------------- #
    #                       Save trajectories informations to csv                  #
    # ---------------------------------------------------------------------------- #

    def print_to_csv(self, csv_file_path, keep_traj = False):
        """Prints all the trajectories into a .csv file in the csv_file_path location """

        if keep_traj == False:
            keep = "n"

        # Check if file .csv exists or not, ask the user to keep it and append to it or replace it
        file_exists = os.path.exists(csv_file_path)
        if (file_exists == False) and (self.ncalls == 0):
            print("---")
            print(
                "The 📄.csv file for the trajectories data doesn't exists, creating it..."
            )
            open(csv_file_path, "a")
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
                open(csv_file_path, "a")
            print("---")
            #self.ncalls = 1

        csv_f = open(csv_file_path, 'a')
        writer = csv.writer(csv_f)
        
        header = ['x', 'y', 'f']
        trajectory = np.stack((self.x_traj_tot, self.y_traj_tot, self.f_ref_tot), axis=-1)

        writer.writerow(header)
        writer.writerows(trajectory)

        csv_f.close()