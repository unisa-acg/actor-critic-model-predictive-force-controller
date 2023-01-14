import os
import robosuite as suite
from robosuite.controllers import load_controller_config
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib
from traj_gen import TrajectoryGenerator
from traj_resampler import TrajectoryResampler
from data_processing import DataProcessing
from sim_data_utilities import SimulationDataUtilities
from data_for_dataset import DataForDataset
from force_controller import ForceController
from datetime import datetime
from torchensemble.utils import io
from torchensemble import (FusionRegressor, GradientBoostingRegressor,
                           SnapshotEnsembleRegressor, VotingRegressor,
                           AdversarialTrainingRegressor)
import nn_utilities_bit as nn_utils
import torch
from force_controller import ForceController
import plot_utils

matplotlib.use('QtCairo')


# Function to customize the environment
def modify_XML():
    tree = ET.parse('sim_model.xml')
    root = tree.getroot()

    for body in root.iter('body'):
        if body.attrib['name'] == 'gripper0_eef':
            attrib_geom = {
                'name': 'ee_sphere_visual',
                'size': '0.4 0.4 0.4',
                'type': "sphere",
                'solref': "0.01 1",
                'contype': "0",
                'conaffinity': "0",
                'group': "1",
                'rgba': "1 1 1 1",
                'mesh': "robot0_link7_vis"
            }
            geom_to_add = ET.Element('geom', attrib_geom)
            attrib_geom2 = {
                'name': 'ee_sphere_collision',
                'size': '0.4 0.4 0.4',
                'type': "sphere",
                'rgba': "1 1 1 1",
                'mesh': "robot0_link7"
            }
            geom2_to_add = ET.Element('geom', attrib_geom2)
            attrib_body = {'name': 'ee_sphere', 'pos': '-0.01 -0.009 -0.07'}
            body_to_add = ET.Element('body', attrib_body)
            body_to_add.append(geom_to_add)
            body_to_add.append(geom2_to_add)
            body.append(body_to_add)

    xml_string = ET.tostring(root, encoding='utf8').decode('utf8')
    tree.write('sim_model_mod.xml')

    return xml_string


# Function to add the boundary marker in the renderization
def add_markers_op_zone(env, params_randomizer, table_pos_z):
    #bl : bottom left point
    #tr : top right point
    bl = params_randomizer['operating_zone_points'][0]
    tr = params_randomizer['operating_zone_points'][1]
    pos_marker_bl = np.array([bl[0], bl[1], table_pos_z + 0.1])
    env.viewer.viewer.add_marker(pos=pos_marker_bl,
                                 type=2,
                                 label="bl",
                                 size=np.array([0.01, 0.01, 0.01]),
                                 rgba=np.array([0.5, 0, 0.5, 1.0]))

    pos_marker_tr = np.array([tr[0], tr[1], table_pos_z + 0.1])
    env.viewer.viewer.add_marker(pos=pos_marker_tr,
                                 type=2,
                                 label="tr",
                                 size=np.array([0.01, 0.01, 0.01]),
                                 rgba=np.array([0, 0.5, 0.5, 1.0]))


controller_config = load_controller_config(default_controller="OSC_POSE")
controller_config['control_delta'] = False
controller_config['impedance_mode'] = 'variable_kp'

# create an environment to visualize on-screen
env = suite.make(
    "Lift",
    robots=["Panda"],  # load a Sawyer robot and a Panda robot
    gripper_types=None,  # use default grippers per robot arm #"WipingGripper"
    controller_configs=controller_config,  # each arm is controlled using OSC
    env_configuration="single-arm-opposed",  # (two-arm envs only) arms face each other
    has_renderer=False,  # on-screen rendering
    render_camera="frontview",  # visualize the "frontview" camera
    has_offscreen_renderer=False,  # no off-screen rendering
    control_freq=100,  # 20 hz control for applied actions
    horizon=500000,  # each episode terminates after 200 steps
    use_object_obs=False,  # no observations needed
    use_camera_obs=False,  # no observations needed
)

suite.utils.mjcf_utils.save_sim_model(env.sim, 'sim_model.xml')
xml_string = modify_XML()
env.reset_from_xml_string(xml_string)  # Reset the environment with the desired
# configuration

####

## Selection of the simulation parameters
ee_pos = env._eef_xpos  # End-effector position (point from which the
# approach trajectory starts)
table_pos = env.sim.model.body_pos[1]  # Table position (point in which the
# approach trajectory finishes)
gap = 0.07  # Gap added to table position to ensure minimal penetration
old_time_step = 0.02  # Time step of the generated trajectory
new_time_step = 0.01  # Time step of the resampled trajectory (100Hz)
now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Impedance controller gains
kp_pos = 7500
kp_ori = 5000
kp_pos_z = 2200

# Direct force controller gains
K_f = 2e-4  # Proportional gain
KI_f = 5e-3  # Integral gain

# ORACLE parameters: set delta and the regularizers (alpha and beta)
delta = 0.1
alpha = 2e4
beta = 1e2
num_step = 5000  # Decides how densely the neighborhood is sampled

## Classes instantiation
FC = ForceController(K_f, KI_f)
TG = TrajectoryGenerator(old_time_step)
DP = DataProcessing()
TR = TrajectoryResampler()
DP = DataProcessing()

# Load normalization data, therefore the mean and the standard deviation
csv_file_norm_path = 'example_data/norm_data_example.csv'
_, norm_data = nn_utils.read_data_from_csv(csv_file_norm_path)

mean = norm_data[0, :]
std = norm_data[1, :]

# Select the device to make the NN inference between "cuda" (gpu) and "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create single NN model
num_hidden_layers = 3  # Number of hidden layers
single_nn = nn_utils.NeuralNetwork(
    dropout_prob=0.2,  # Dropout probability 
    num_inputs=2,  # Number of inputs
    num_outputs=1,  # Number of outputs
    num_hidden_neurons=200,  # Number of neurons
    num_hidden_layers=num_hidden_layers,
)
# Create NNs ensemble model, starting from the single one
type = "fusion"
model = FusionRegressor(estimator=single_nn, n_estimators=3, cuda=True)
# print(model)  # Uncomment to print the model

# Load the weights of the pretrained NNs-E on the new model
io.load(model, 'example_data')

# APPROACHING TRAJECTORY
# Parameters definition
waypoints = [(ee_pos[0], ee_pos[2]), (table_pos[0], table_pos[2] + gap + 0.05),
             (table_pos[0], table_pos[2] + gap)]
traj_types = ['line', 'line']
traj_params = [None, None]
traj_timestamps = [0, 2, 6]
force_reference_types = ['cnst', 'cnst']
force_reference_parameters = [0, 0]

# Trajectory generation
[traj_app_x, traj_app_z,
 traj_app_f] = TG.traj_gen(waypoints, traj_types, traj_params, traj_timestamps,
                           force_reference_types, force_reference_parameters)

csv_dir_path = 'output/traj_generated_csv/gen_{}'.format(now)
csv_name = 'traj_gen_{}.csv'
TG.print_to_csv(csv_name, csv_dir_path)

time_vect = np.arange(traj_timestamps[0], traj_timestamps[-1], old_time_step)

traj_matrix = TR.read_traj_from_csv(os.path.join(csv_dir_path, csv_name))

# Trajectory resampler
TR.interp_traj(traj_matrix, time_vect, new_time_step)
csv_res_path = 'output/traj_resampled_csv/res_{}'.format(now)
csv_res_name = 'traj_res_{}.csv'
TR.traj_res_csv(csv_res_name, csv_res_path)
df_approach = TR.read_traj_from_csv(os.path.join(csv_res_path, csv_res_name))

# MAIN TRAJECTORY
# Parameters definition
waypoints = [(table_pos[0], table_pos[1]), (-0.025, 0), (-0.05, 0),
             (table_pos[0], table_pos[1])]
traj_types = ['line', 'line', 'sine_curve']
traj_params = [None, None, [0.05, 1]]
traj_timestamps = [0, 5, 10, 20]
force_reference_types = ['ramp', 'cnst', 'sine_curve']
force_reference_parameters = [[25, 35], 35, [35, 15, 10, 10]]

# Defining the operational zone boundaries along the xy-plane
params_randomizer = {
    "operating_zone_points": [(-0.25, -0.25), (0.25, 0.25)]
}  # The first is y, the second x

# Trajectory generation
[x, y, f] = TG.traj_gen(waypoints, traj_types, traj_params, traj_timestamps,
                        force_reference_types, force_reference_parameters)
# Overwrite the approach trajectory .csv files
TG.print_to_csv(csv_name, csv_dir_path)

fig = TG.plot_traj(x, y, params_randomizer)  # Plot the generated trajectory
fig2 = TG.plot_force_ref(f, traj_timestamps)  # Plot the resampled trajectory

time_vect = np.arange(traj_timestamps[0], traj_timestamps[-1], old_time_step)
traj_matrix = TR.read_traj_from_csv(os.path.join(csv_dir_path, csv_name))

# Trajectory resampler
TR.interp_traj(traj_matrix, time_vect, new_time_step)
TR.traj_res_csv(csv_res_name, csv_res_path)
dframe = TR.read_traj_from_csv(os.path.join(csv_res_path, csv_res_name))

# Vectors to traform the 2D trajectories in 3D
traj_app_y = np.ones(len(df_approach)) * ee_pos[1]
table_z = np.ones(len(dframe)) * (table_pos[2] + gap)

# Force vector given to the approach trajectory (it is not used as there is no contact)
force_app = np.ones(len(df_approach)) * np.array(dframe[:, 2], dtype=np.float64)[0]

# Variables initialization
dt = new_time_step
pos_to_check_old = 0
flag_contact = 0
flag_ORACLE_activation = 0
# Vectors for plots
step = np.arange(len(force_app), len(force_app) + len(dframe[:, 2]), 1) * dt
i_dis = []
steps = []

xd_vect_for_err_plot = []
xc_vect_for_plot = []
xd_vect_for_plot = []
xf_vect_for_plot = []
xf_vect_for_plot_input = []

force_sensor = []

sub_error = []
error = []
error_NN = []
vel_NN = []

# Creation of a buffer to apply a low filter to the contact forces
smooth_ft_buffer = []  #deque(maxlen=20)
smooth_ft_buffer.append(np.array([0, 0, 0, 0, 0, 0]))

# Modification of the tau of the environment (stiffness and damping)
env.sim.model.geom_solref[7][0] = 0.2
env.sim.model.geom_solref[30][0] = 0.2

# Modification of frictions
env.sim.model.geom_friction[7] = [2e-3, 0.005, 0.0001]  #default[1.0, 0.005, 0.0001]
env.sim.model.geom_friction[30] = [2e-3, 0.005, 0.0001]

# TRAJECTORY BUILDING
# Approaching trajectory
traj_approach = np.stack(
    (
        np.array(df_approach[:, 0], dtype=np.float64),
        traj_app_y,
        np.array(df_approach[:, 1], dtype=np.float64),
    ),
    axis=-1,
)
# Main trajectory
main_traj = np.stack(
    (
        np.array(dframe[:, 0], dtype=np.float64),
        np.array(dframe[:, 1], dtype=np.float64),
        table_z,
    ),
    axis=-1,
)
# Final trajectory
target_traj = np.concatenate((traj_approach, main_traj))
# Final reference force
force_ref = np.concatenate((force_app, np.array(dframe[:, 2], dtype=np.float64)))

# Classes instantiation (useful for plots)
DFD = DataForDataset(len(target_traj))
SDU = SimulationDataUtilities(env.horizon)

### TRAJECTORY EXECUTION
for ii in range(len(target_traj)):

    # Force filtering
    smooth_ft_buffer.append(np.array(env.sim.data.sensordata))
    if len(smooth_ft_buffer) > 2:
        smooth_ft_buffer.pop(0)
    force_vals = np.mean(np.asarray(smooth_ft_buffer).copy(), 0)

    # Stabilization of the contact with the environment
    if ii == len(df_approach) + 100:
        flag_contact = 1
        kk = ii

    # Activation of the force controller and ORACLE
    if ii == len(df_approach) + 150:
        flag_ORACLE_activation = 1
        k = ii

    if flag_contact == 1:
        f_d = force_ref[ii]  # Force desired
        f_e = force_vals[2]  # Actual force (filtered)

        # Direct force controller
        x_f = FC.force_controller(f_d, f_e, dt)

        ## ORACLE
        # Build the neighborhood
        min_pos = x_f - delta
        max_pos = x_f + delta
        pos_to_check = np.linspace(min_pos, max_pos, num_step)
        if pos_to_check_old == 0:
            vel_to_check = (pos_to_check - x_f) / dt
        else:
            vel_to_check = (pos_to_check - x_d) / dt

        # Denormalize the vectors for the inference
        pos_input = (pos_to_check - mean[2]) / std[2]
        vel_input = (vel_to_check - mean[8]) / std[8]

        # Build the vector for the inference
        posvel_to_check = torch.from_numpy(np.stack((pos_input, vel_input),
                                                    axis=-1)).to(device).float()
        # Make inference on the NNs-E
        model.eval()
        with torch.no_grad():
            force_to_check_norm = model.forward(posvel_to_check).cpu().numpy()

        force_to_check = force_to_check_norm * std[20] + mean[20]

        x_c_possible = pos_to_check - x_f  # Vector of possible x_c

        # BUILD THE COST FUNCTION J
        mse = np.abs((f_d - force_to_check))  # first term: |f_d - f_env|
        alpha_term = alpha * np.power(x_c_possible, 2)  # second term: alpha*x_c^2
        if pos_to_check_old == 0:  # third term active only after the first iteration
            beta_term = np.zeros(len(mse))
        else:
            # beta*|x_c - x_c_old|
            beta_term = beta * np.abs(pos_to_check - pos_to_check_old)
        # Cost function
        J = mse + np.expand_dims(alpha_term, axis=-1) + np.expand_dims(beta_term,
                                                                       axis=-1)
        idx_xc = np.argmin(J)  # Index at wich J has the minimum
        x_c = x_c_possible[idx_xc]  # Correct position x_c

        x_d = x_f + x_c  # Desired setpoint sent to the impedance controller

        pos_to_check_old = pos_to_check[idx_xc]

    ## LOW-LEVEL IMPEDANCE CONTROLLER
    # Build and select the inputs (position and parameters)
    if flag_contact == 1:
        traj = np.array(
            [target_traj[ii, 0], target_traj[ii, 1], target_traj[ii, 2] + x_d])
    else:
        traj = np.array(target_traj[ii, :])

    KP = np.array([kp_pos, kp_pos, kp_pos_z, kp_ori, kp_ori, kp_ori])
    pose = np.append(traj, np.array([0, -np.pi, 0]))
    action = np.append(KP, pose)
    env.step(action)  # Action of the step sent to the impedance controller

    # PLOTS VARIABLES SAVING
    SDU.get_info_robot(env, 'gripper0_grip_site')
    force_sensor.append(force_vals[2])
    i_dis.append(ii * dt)

    if flag_contact == 1:  # After contact
        steps.append(ii * dt)
        xd_vect_for_plot.append(x_f + x_c)
        xc_vect_for_plot.append(x_c)
        xf_vect_for_plot.append(x_f)
        DFD.get_info_robot(env, 'gripper0_grip_site', f_e, x_f)

    if flag_ORACLE_activation == 1:  # Once the force controller is activated
        error.append(f_d - env.sim.data.sensordata[2])
        error_NN.append(force_to_check[idx_xc] - env.sim.data.sensordata[2])
        sub_error.append(abs(f_d - force_to_check[idx_xc]))
        xd_vect_for_err_plot.append(x_f + x_c)
        vel_NN.append(vel_to_check[idx_xc])

    ## Uncomment to visualize the rendering
    # add_markers_op_zone(env, params_randomizer, table_pos[2])
    # env.render()

# After each trajectory an environment reset is done to restart from the neutral pose
env.reset_from_xml_string(xml_string)

# Calculation of the force tracking error
mse = (np.power(error, 2)).mean()
print(f'The real mean square error is {mse}')

mse_NN = (np.power(error_NN, 2)).mean()
print(f'The mean square error of the prediction is {mse_NN}')

max_xf = max(xd_vect_for_err_plot)
# 3D plot of the real error
plot_utils.plot_explored_space_3D(xd_vect_for_err_plot - max_xf,
                                  vel_NN,
                                  error,
                                  elev=90,
                                  azim=-90)
# Plot of the real error along time
plot_utils.plot_dataset_along_time(xd_vect_for_err_plot - max_xf, vel_NN, error)

# 3D plot of the prediction error
plot_utils.plot_explored_space_3D(xd_vect_for_err_plot - max_xf,
                                  vel_NN,
                                  sub_error,
                                  elev=90,
                                  azim=-90,
                                  vmin=0)
# Plot of the real error along time
plot_utils.plot_dataset_along_time(xd_vect_for_err_plot - max_xf, vel_NN, sub_error)

SDU.plot_torques()
SDU.plot_ee_pos_vel()

plt.figure()
plt.axvline(x=kk * dt, color='green', linestyle='dashed')
plt.plot(step[100:], dframe[100:, 2], color='#005B96')
plt.plot(i_dis, force_sensor, color='#ba0001')
plt.legend(['ORACLE activation', 'Reference force', 'Actual force'])
plt.xlabel('$Time$')
plt.ylabel('$Force \, [N]$')
plt.grid()

plt.figure()
plt.plot(steps, xf_vect_for_plot, 'r')
plt.plot(steps, xc_vect_for_plot)
plt.plot(steps, xd_vect_for_plot, 'b')
plt.legend(['z_f', 'z_c', 'z_f+z_c'])
plt.grid()

plt.show()