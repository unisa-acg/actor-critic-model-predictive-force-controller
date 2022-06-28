from cmath import sqrt
import numpy as np
import matplotlib.pyplot as plt
from traj_gen import TrajectoryGenerator
from traj_resampler import TrajectoryResampler
import randomizer as R

TG = TrajectoryGenerator()
T = TrajectoryResampler()

num_traj = 4
old_time_step = 0.1
new_time_step = 0.002

for i in range(num_traj):
    #Trajectory parameters randomization
    params_randomizer = {
        'starting_point': (0, 0),
        "operating_zone_points": [(-0.25, -0.25), (0.25, 0.25)],  # il primo è y il secondo x
        'max_n_subtraj': 5,
        'max_vel': 3,
        'max_radius': 0.1,
        'min_radius': 0.01,
        'max_ampl': 0.1,
        'max_freq': 10,
        'min_f_ref': 10,
        'max_f_ref': 80
    }
    waypoints, traj_timestamps, traj_types, traj_params, force_reference_types, force_reference_parameters = R.traj_randomizer(params_randomizer)

    #Trajectory generation
    [x, y, f] = TG.traj_gen(waypoints, traj_types, traj_params, traj_timestamps,
                            force_reference_types, force_reference_parameters)

    TG.print_to_csv('csv_folder/traj_gen_{}.csv'.format(i))

    time = np.arange(traj_timestamps[0], traj_timestamps[-1], old_time_step)

    traj_matrix = T.read_traj_from_csv('csv_folder/traj_gen_{}.csv'.format(i))

    #Trajectory resampling
    T.interp_traj(traj_matrix, time, new_time_step)
    T.traj_res_csv('csv_folder/traj_res_{}.csv'.format(i))
    traj_res = T.read_traj_from_csv('csv_folder/traj_res_{}.csv'.format(i))

    t_dis = np.arange(traj_timestamps[0], traj_timestamps[-1] - old_time_step, new_time_step)

    #Trajectory plotting
    fig = TG.plot_traj(x, y, traj_timestamps, params_randomizer)

    fig2 = TG.plot_force_ref(f, traj_timestamps)

    plt.figure(1+2*i)
    plt.plot(traj_res[:,0], traj_res[:,1], label='Resampled')
    plt.legend()

    plt.figure(2+2*i)
    plt.plot(t_dis, traj_res[:,2], label='Resampled')
    plt.legend()

plt.show()