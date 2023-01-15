import matplotlib.pyplot as plt
import numpy as np
import os
from trajectories_generation_utilities.src.traj_gen import TrajectoryGenerator
from trajectories_generation_utilities.src.traj_resampler import TrajectoryResampler

num_traj = 3
old_time_step = 0.1
new_time_step = 0.002

TG = TrajectoryGenerator(old_time_step)
TR = TrajectoryResampler()

for i in range(num_traj):
    # Trajectory generation
    # Trajectory data
    waypoints = [(0, 0), (0, 1), (0, 1), (1, 1), (2, 2)]
    traj_types = ["line", "circle", "line", "sine_curve"]
    traj_params = [None, 0.2, None, [1, 3]]
    traj_timestamps = [0, 3, 4, 4.5, 6.5]
    force_reference_types = ["cnst", "cnst", "ramp", "ramp"]
    force_reference_parameters = [0, 1, [1, 3], [3, 0]]

    [x, y, f] = TG.traj_gen(
        waypoints,
        traj_types,
        traj_params,
        traj_timestamps,
        force_reference_types,
        force_reference_parameters,
    )

    csv_dir_path = 'output/traj_generated_csv'
    csv_name = 'traj_gen.csv'
    TG.print_to_csv(csv_name, csv_dir_path)

    time_vect = np.arange(traj_timestamps[0], traj_timestamps[-1], old_time_step)

    traj_matrix = TR.read_traj_from_csv(os.path.join(csv_dir_path, csv_name))

    # Trajectory resampling
    TR.interp_traj(traj_matrix, time_vect, new_time_step)
    csv_res_path = 'output/traj_resampled_csv'
    csv_res_name = 'traj_res.csv'
    TR.traj_res_csv(csv_res_name, csv_res_path)
    traj_res = TR.read_traj_from_csv(os.path.join(csv_res_path, csv_res_name))

    t_dis = np.arange(traj_timestamps[0], traj_timestamps[-1] - old_time_step,
                      new_time_step)
    # Defining the operational zone boundaries along the xy-plane
    params_randomizer = {
        "operating_zone_points": [(-1, -1), (3, 3)]
    }  # The first is y, the second x

    # Trajectory plotting
    fig = TG.plot_traj(x, y, params_randomizer)  # Plot the generated trajectory
    fig2 = TG.plot_force_ref(f, traj_timestamps)  # Plot the resampled trajectory

    plt.figure(1 + 2 * i)
    plt.plot(traj_res[:, 0], traj_res[:, 1], label="Resampled")
    plt.legend()

    plt.figure(2 + 2 * i)
    plt.plot(t_dis, traj_res[:, 2], label="Resampled")
    plt.legend()

plt.show()
