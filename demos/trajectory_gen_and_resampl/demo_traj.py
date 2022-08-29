import matplotlib.pyplot as plt
import numpy as np

from trajectories_generation.src.traj_gen import TrajectoryGenerator
from trajectories_generation.src.traj_resampler import TrajectoryResampler

TG = TrajectoryGenerator()
T = TrajectoryResampler()

num_traj = 3
old_time_step = 0.1
new_time_step = 0.002

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

TG.print_to_csv(
    num_traj,
    waypoints,
    traj_types,
    traj_params,
    traj_timestamps,
    force_reference_types,
    force_reference_parameters,
)

time = np.arange(traj_timestamps[0], traj_timestamps[-1], old_time_step)

[df, traj_count] = T.read_traj_from_csv("traj_gen.csv")

for i in range(traj_count):
    traj_matrix = np.stack(
        (
            np.array(df["x" + str(i)], dtype=np.float64),
            np.array(df["y" + str(i)], dtype=np.float64),
            np.array(df["f" + str(i)], dtype=np.float64),
        ),
        axis=-1,
    )

# Trajectory resampling
T.traj_res_csv(traj_count, traj_matrix, time, new_time_step)
[dframe, traj_count] = T.read_traj_from_csv("traj_resampled.csv")

t_dis = np.arange(
    traj_timestamps[0], traj_timestamps[-1] - old_time_step, new_time_step
)

for i in range(traj_count):
    traj_res = np.stack(
        (
            np.array(dframe["x" + str(i)], dtype=np.float64),
            np.array(dframe["y" + str(i)], dtype=np.float64),
            np.array(dframe["f" + str(i)], dtype=np.float64),
        ),
        axis=-1,
    )

    # Trajectory plotting
    fig = TG.plot_traj(x, y, traj_timestamps)
    fig2 = TG.plot_force_ref(f, traj_timestamps)
    plt.figure(1 + 2 * i)
    plt.plot(traj_res[:, 0], traj_res[:, 1])
    plt.figure(2 + 2 * i)
    plt.plot(t_dis, traj_res[:, 2])

plt.show()
