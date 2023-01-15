import matplotlib.pyplot as plt
import nn_utilities as nn_utils
import plot_utils
import numpy as np

csv_files_path = 'output/processed_csv/processed_2023_01_13_19_22_11/sim_data_proc.csv'
csv_files_norm_path = 'output/processed_csv/processed_2023_01_13_19_22_11/norm_data.csv'
fieldnames, robot_info = nn_utils.read_data_from_csv(csv_files_path)
_, norm_data = nn_utils.read_data_from_csv(csv_files_norm_path)


# Function to extract the data position from the dataset
def return_idx_nnz(col_names):
    # return only the indexes of the quantity useful in a z-focused nn
    idx_x_f = col_names.index('pos_z')
    idx_xf_dot = col_names.index('vel_z')
    idx_f_z_out = col_names.index('f_z_out')

    column_idx_dict = {
        'idx_x_f': idx_x_f,
        'idx_xf_dot': idx_xf_dot,
        'idx_f_z_out': idx_f_z_out
    }

    return column_idx_dict


column_idx_dict = return_idx_nnz(fieldnames)
reduced_dataset = robot_info[:, list(column_idx_dict.values())]
xf_norm = reduced_dataset[:, 0][:-1]
xf_dot_norm = reduced_dataset[:, 1][:-1]
fz_out_norm = reduced_dataset[:, 2][:-1]

mean = norm_data[0, :]
std = norm_data[1, :]

x_f = xf_norm * std[2] + mean[2]
xf_dot = xf_dot_norm * std[8] + mean[8]
f_z = fz_out_norm * std[20] + mean[20]

dt = 0.01
step_vect = np.arange(0, len(f_z), 1) * dt

## Plots
plot_utils.distributions_plot([x_f, xf_dot, f_z], bins=20)
plot_utils.plot_dataset_along_time(x_f, xf_dot, f_z)

plt.show()