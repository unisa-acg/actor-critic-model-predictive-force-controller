import robosuite as suite
from robosuite.controllers import load_controller_config
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
import os

import trajectories_generation.src.randomizer as R
from trajectories_generation.src.traj_gen import TrajectoryGenerator
from trajectories_generation.src.traj_resampler import TrajectoryResampler

from dataset_generation.src.dataset_utilities.data_for_dataset import DataForDataset
from dataset_generation.src.dataset_utilities.data_processing import DataProcessing
from dataset_generation.src.utilities.sim_data_utilities import SimulationDataUtilities
from dataset_generation.src.controller.admittance_controller_1d import (
    AdmittanceController1D)

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
    bl = params_randomizer['operating_zone_points'][0]  # bl : bottom left point
    tr = params_randomizer['operating_zone_points'][1]  # tr : top right point
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

# Create an environment to visualize on-screen
env = suite.make(
    "Lift",
    robots=["Panda"],  # Load a Panda robot
    gripper_types=None,  # Not use default grippers per robot arm 
    controller_configs=controller_config,  # The arm is controlled using OSC
    env_configuration="single-arm-opposed",  # Single arm
    has_renderer=True,  # On-screen rendering
    render_camera="frontview",  # Visualize the "frontview" camera
    has_offscreen_renderer=False,  # No off-screen rendering
    control_freq=500,  # 500 hz control for applied actions
    horizon=500000,  # Maximum number of steps for each episode
    use_object_obs=False,  # No observations needed
    use_camera_obs=False,  # No observations needed
)

suite.utils.mjcf_utils.save_sim_model(env.sim, 'sim_model.xml')
xml_string = modify_XML()
env.reset_from_xml_string(xml_string)  # Reset the environment with the desired
# configuration

####

# Admittance controller settings
# Data retrieval
mass_matrix = env.robots[0].controller.mass_matrix
J_pos = env.robots[0].controller.J_pos[2, :]
# Data manipulation
mass_matrix_inv = np.linalg.inv(mass_matrix)
lambda_pos_inv = np.dot(np.dot(J_pos, mass_matrix_inv), J_pos.transpose())
lambda_pos = 1 / lambda_pos_inv
# Controller gains
K_p = 1.5e-4  # Position gain
K_d = 1.5e-3  # Velocity gain
K_f = 8e-2  # Force gain

# Class instantiation
A = AdmittanceController1D(lambda_pos_inv, K_p, K_d, K_f)
TG = TrajectoryGenerator()
TR = TrajectoryResampler()
DP = DataProcessing()
SDU = SimulationDataUtilities(env.horizon)

# Selection of the parameters
num_traj = 1  # Number of trajectories
ee_pos = env._eef_xpos  # End-effector position (point from which the approach
# trajectory starts)
table_pos = env.sim.model.body_pos[1]  # Table position (point in which the approach
# trajectory finishes)
gap = 0.072  # Gap added to table position to ensure minimal penetration
kp_pos = 150  # Gain of the position controller
kp_f = 0.1  # Gain of the force controller
old_time_step = 0.1  # Time step of the generated trajectory
new_time_step = 0.002  # Time step of the resampled trajectory (500Hz)

now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

for i in range(num_traj):
    # APPROACHING TRAJECTORY
    # Parameters definition
    waypoints = [(ee_pos[0], ee_pos[2]), (table_pos[0], table_pos[2] + gap + 0.05),
                 (table_pos[0], table_pos[2] + gap)]
    traj_types = ['line', 'line']
    traj_params = [None, None]
    traj_timestamps = [0, 1, 5]
    force_reference_types = ['cnst', 'cnst']
    force_reference_parameters = [0, 0]

    # Trajectory generation
    [traj_app_x, traj_app_z,
     traj_app_f] = TG.traj_gen(waypoints, traj_types, traj_params, traj_timestamps,
                               force_reference_types, force_reference_parameters)

    csv_dir_path = 'output/traj_generated_csv/gen_{}'.format(now)
    csv_name = 'traj_gen_{}.csv'.format(i)
    TG.print_to_csv(csv_name, csv_dir_path)

    time_vect = np.arange(traj_timestamps[0], traj_timestamps[-1], old_time_step)

    traj_matrix = TR.read_traj_from_csv(os.path.join(csv_dir_path, csv_name))

    # Trajectory resampler
    TR.interp_traj(traj_matrix, time_vect, new_time_step)
    csv_res_path = 'output/traj_resampled_csv/res_{}'.format(now)
    csv_res_name = 'traj_res_{}.csv'.format(i)
    TR.traj_res_csv(csv_res_name, csv_res_path)
    df_approach = TR.read_traj_from_csv(os.path.join(csv_res_path, csv_res_name))

    # MAIN TRAJECTORY
    # Randomization of the parameters
    steps_required = 1e6
    while steps_required > 1e5:  # Maximum steps ammited
        params_randomizer = {
            'starting_point': (0, 0),
            "operating_zone_points": [(-0.25, -0.25),
                                      (0.25, 0.25)],  # il primo è y il secondo x
            'max_n_subtraj': 4,
            'max_vel': 0.02,
            'max_radius': 0.1,
            'min_radius': 0.01,
            'max_ampl': 0.1,
            'max_freq': 10,
            'min_f_ref': 10,
            'max_f_ref': 80,
            'max_ampl_f': 20,
            'max_freq_f': 10,
        }

        # Parameters definition
        [
            waypoints, traj_timestamps, traj_types, traj_params, force_reference_types,
            force_reference_parameters
        ] = R.traj_randomizer(params_randomizer)

        steps_required = (traj_timestamps[-1] + 6) / 0.002
    print('Steps required: ', steps_required)

    # Trajectory generation
    [x, y, f] = TG.traj_gen(waypoints, traj_types, traj_params, traj_timestamps,
                            force_reference_types, force_reference_parameters)
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

    # Force vector given to the approach trajectory (it is not used as there is no
    # contact)
    force_app = np.ones(len(df_approach)) * np.array(dframe[:, 2], dtype=np.float64)[0]

    # Variables initialization
    i_dis = []
    flag = 0
    flag_dataset = 0
    force_sensor = []

    # Creation of a buffer to apply a low filter to the contact forces
    smooth_ft_buffer = []
    smooth_ft_buffer.append(np.array([0, 0, 0, 0, 0, 0]))

    # Modification of the table stiffness keeping it critically damped
    env.sim.model.geom_solref[7][0] = 0.1
    # Modification of frictions
    env.sim.model.geom_friction[7] = [0, 0, 0]
    env.sim.model.geom_friction[30] = [0, 0, 0]

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

    # Last class instantiation
    DFD = DataForDataset(len(target_traj))

    # Trajectory execution
    for ii in range(len(target_traj)):
        SDU.contact_forces(env, 'table_collision', 'ee_sphere_collision')

        # Force smoothing
        smooth_ft_buffer.append(np.array(env.sim.data.sensordata))
        if len(smooth_ft_buffer) > 5:
            smooth_ft_buffer.pop(0)
        force_vals = np.mean(np.asarray(smooth_ft_buffer).copy(), 0)

        # Activation of the admittance controller only when the end-effector impact the
        # table
        if env.check_contact('ee_sphere_collision', 'table_collision'):
            flag = 1

        if flag == 1:
            f_d = force_ref[ii]  # Reference force
            x_d = target_traj[ii, 2]  # Reference position
            A.set_reference(x_d, f_d)  # Setpoint for u_h in form [x_d, f_d]

            # Actual position of the end-effector along z-axis
            pos_along_z = env.sim.data.site_xpos[env.sim.model.site_name2id(
                'gripper0_grip_site')][2]
            # Actual velocity of the end-effector along z-axis
            vel_along_z = env.sim.data.site_xvelp[env.sim.model.site_name2id(
                'gripper0_grip_site')][2]
            # Next z position identified by the admittance controller
            u_h = A.update(force_vals[2], (pos_along_z, vel_along_z), new_time_step)

        # Effective target trajectory
        # After the contact the target position along z, given to the position
        # controller, is selected by the admittance controller (u_h)
        if flag == 1:
            traj = np.array([target_traj[ii, 0], target_traj[ii, 1], u_h])
            kp_pos = 150 * 1e4
            KP = np.array([150, 150, kp_pos, 150, 150, 150])
        else:
            traj = np.array(target_traj[ii, :])
            kp_pos = 150
            KP = np.array([150, 150, kp_pos, 150, 150, 150])

        pose = np.append(traj, np.array([0, -np.pi, 0]))
        action = np.append(KP, pose)
        env.step(
            action)  # Action performed by the position controller at the actual step

        # Contact information recording for the final dataset

        # The following condition allows to avoid the recording of the oscillating data
        # coming from the impact during the approaching trajectory
        if ii == len(df_approach) + 600:
            flag_dataset = 1
            k = ii
        if flag_dataset == 1:
            DFD.get_info_robot(env, 'gripper0_grip_site')

        # Saving of the smoothed force
        force_sensor.append(force_vals[2])
        i_dis.append(ii)

        SDU.get_info_robot(
            env, 'gripper0_grip_site')  # Contact information recording for the plots

        add_markers_op_zone(env, params_randomizer, table_pos[2])

        # env.render()

    # After each trajectory an environment reset is done to restart from the neutral
    # pose
    env.reset_from_xml_string(xml_string)

    # Output generation
    SDU.plot_contact_forces()
    SDU.plot_torques()
    SDU.plot_ee_pos_vel()
    csv_unproc_path = 'output/unprocessed_csv/unprocessed_{}'.format(now)
    csv_unproc_name = 'sim_data_{}.csv'.format(i)
    DFD.contact_info_to_csv(csv_unproc_name,
                            csv_unproc_path)  # Unprocessed data stored in csv file

    # Smoothed force plotted
    plt.figure()
    plt.axvline(x=k, linestyle='dashed', color='red')
    plt.plot(i_dis, force_sensor, color='#125488')
    plt.legend(['Step when recording starts'])
    plt.xlabel('steps')
    plt.ylabel('[N]')
    plt.title('Smoothed force from sensor')
    plt.grid()

DP.combine_trajectory_csv(csv_unproc_path, False)
csv_proc_path = 'output/processed_csv/processed_{}'.format(now)
csv_proc_name = 'sim_data_proc.csv'
csv_norm_data_name = 'norm_data.csv'  # data for the dataset normalization and viceversa
DP.contact_info_processed_to_csv(csv_proc_name, csv_proc_path,
                                 csv_norm_data_name)  # Creation of the dataset
DP.plot_data_comparison()
plt.show()
