import os
import numpy as np
import matplotlib.pyplot as plt
from trajectories_generation.src.traj_gen import TrajectoryGenerator
from trajectories_generation.src.traj_resampler import TrajectoryResampler
import trajectories_generation.src.randomizer as R
import os
from datetime import datetime

TG = TrajectoryGenerator()
TR = TrajectoryResampler()

num_traj = 4  #number of trajectories
old_time_step = 0.1  #time step of the generated trajectories
new_time_step = 0.002  #time step of the resampled trajectories

for i in range(num_traj):
    #Trajectory parameters randomization
    params_randomizer = {
        'starting_point': (0, 0),
        "operating_zone_points": [(-0.25, -0.25), (0.25, 0.25)
                                 ],  # first element of tuple is y, the second one x
        'max_n_subtraj': 5,
        'max_vel': 3,
        'max_radius': 0.1,
        'min_radius': 0.01,
        'max_ampl': 0.1,
        'max_freq': 10,
        'min_f_ref': 10,
        'max_f_ref': 80,
        'max_ampl_f': 20,
        'max_freq_f': 10,
    }
    [
        waypoints, traj_timestamps, traj_types, traj_params, force_reference_types,
        force_reference_parameters
    ] = R.traj_randomizer(params_randomizer)

    #Trajectory generation
    [x, y, f] = TG.traj_gen(waypoints, traj_types, traj_params, traj_timestamps,
                            force_reference_types, force_reference_parameters)

    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    csv_dir_path = '../output/example_traj_{}'.format(now)
    csv_name = 'traj_gen_{}.csv'.format(i)
    TG.print_to_csv(csv_name, csv_dir_path)

    time_vect = np.arange(traj_timestamps[0], traj_timestamps[-1], old_time_step)

    traj_matrix = TR.read_traj_from_csv(os.path.join(csv_dir_path, csv_name))

    #Trajectory resampling
    TR.interp_traj(traj_matrix, time_vect, new_time_step)
    csv_res_name = 'traj_res_{}.csv'.format(i)
    csv_res_path = os.path.join(csv_dir_path, csv_res_name)
    TR.traj_res_csv(csv_res_path)
    traj_res = TR.read_traj_from_csv(csv_res_path)

    t_dis = np.linspace(traj_timestamps[0], traj_timestamps[-1] - old_time_step,
                        len(traj_res))

    #Trajectory plotting
    fig = TG.plot_traj(x, y, params_randomizer)

    fig2 = TG.plot_force_ref(f, traj_timestamps)

    plt.figure(1 + 2 * i)
    plt.plot(traj_res[:, 0], traj_res[:, 1], label='Resampled')
    plt.legend()

    plt.figure(2 + 2 * i)
    plt.plot(t_dis, traj_res[:, 2], label='Resampled')
    plt.legend()

plt.show()