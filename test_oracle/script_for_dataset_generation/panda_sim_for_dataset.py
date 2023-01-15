import os
import robosuite as suite
from robosuite.controllers import load_controller_config
import numpy as np
from datetime import datetime
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib
from trajectories_generation_utilities.src.traj_gen import TrajectoryGenerator
from trajectories_generation_utilities.src.traj_resampler import TrajectoryResampler
import trajectories_generation_utilities.src.randomizer as R
from dataset_generation_utilities.src.dataset_utilities.data_processing import DataProcessing
from dataset_generation_utilities.src.dataset_utilities.data_for_dataset import DataForDataset
from dataset_generation_utilities.src.PI_force_controller.force_controller import ForceController
from utilities.src.sim_data_utilities import SimulationDataUtilities

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
    robots=["Panda"],  # load a Panda robot
    gripper_types=None,  # use not default grippers per robot arm
    controller_configs=controller_config,  # the arm is controlled using OSC
    env_configuration="single-arm-opposed",  # single arm env
    has_renderer=False,  # on-screen rendering
    render_camera="frontview",  # visualize the "frontview" camera
    has_offscreen_renderer=False,  # no off-screen rendering
    control_freq=100,  # 100 hz control for applied actions
    horizon=500000,  # maximum number of steps of each episode
    use_object_obs=False,  # no observations needed
    use_camera_obs=False,  # no observations needed
)

suite.utils.mjcf_utils.save_sim_model(env.sim, 'sim_model.xml')
xml_string = modify_XML()
env.reset_from_xml_string(xml_string)  # Reset the environment with the desired
# configuration

####

## Selection of the simulation parameters
num_traj = 2  # Number of trajectories
ee_pos = env._eef_xpos  # End-effector position (point from which the
# approach trajectory starts)
table_pos = env.sim.model.body_pos[1]  # Table position (point in which the
# approach trajectory finishes)
gap = 0.07  # Gap added to table position to ensure minimal
# penetration
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

## Class instantiation
FC = ForceController(K_f, KI_f)
TG = TrajectoryGenerator(old_time_step)
TR = TrajectoryResampler()
DP = DataProcessing()

## Trajectories generation
for i in range(num_traj):
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
    # Randomization of the parameters
    steps_required = 1e6
    while steps_required > 1e4:  # Maximum steps ammited
        params_randomizer = {
            'starting_point': (0, 0),
            "operating_zone_points": [(-0.25, -0.25), (0.25, 0.25)],
            'max_n_subtraj': 10,
            'max_vel': 0.025,
            'max_radius': 0.1,
            'min_radius': 0.01,
            'max_ampl': 0.1,
            'max_freq': 10,
            'min_f_ref': 10,
            'max_f_ref': 50,
            'max_ampl_f': 10,
            'max_freq_f': 5,
        }
        # Parameters definition
        [
            waypoints, traj_timestamps, traj_types, traj_params, force_reference_types,
            force_reference_parameters
        ] = R.traj_randomizer(params_randomizer)

        steps_required = (traj_timestamps[-1] + 6) / 0.002
    print('Trajectory ', i + 1, '--> steps required: ', steps_required)

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
    flag_dataset = 0
    flag = 0
    # Vectors for plots
    step = np.arange(len(force_app), len(force_app) + len(dframe[:, 2]), 1) * dt
    steps = []
    force_sensor = []

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

    # Classes instantiation
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
        if ii == len(df_approach) + 50:
            flag = 1
            k = ii

        # Start the data recording
        if ii == len(df_approach) + 100:
            flag_dataset = 1
            kk = ii

        if flag == 1:
            f_d = force_ref[ii]
            f_e = force_vals[2]  #env.sim.data.sensordata[2]

            #Force controller
            x_f = FC.force_controller(f_d, f_e, dt)

        ## LOW-LEVEL IMPEDANCE CONTROLLER
        # Build and select the inputs (position and parameters)
        if flag == 1:
            traj = np.array(
                [target_traj[ii, 0], target_traj[ii, 1], target_traj[ii, 2] + x_f])
        else:
            traj = np.array(target_traj[ii, :])

        KP = np.array([kp_pos, kp_pos, kp_pos_z, kp_ori, kp_ori, kp_ori])
        pose = np.append(traj, np.array([0, -np.pi, 0]))
        action = np.append(KP, pose)
        env.step(action)  # Action of the step sent to the impedance controller

        # PLOTS VARIABLES SAVING
        SDU.get_info_robot(env, 'gripper0_grip_site')
        SDU.contact_forces(env, 'table_collision', 'ee_sphere_collision')
        steps.append(ii * dt)
        force_sensor.append(env.sim.data.sensordata[2])

        if flag_dataset == 1:  # After contact
            DFD.get_info_robot(env, 'gripper0_grip_site', f_e, x_f)

        ## To visualize the rendering
        if env.has_renderer:
            add_markers_op_zone(env, params_randomizer, table_pos[2])
            env.render()

    # After each trajectory an environment reset is done to restart from the neutral pose
    env.reset_from_xml_string(xml_string)

    # Output generation
    SDU.plot_contact_forces()
    SDU.plot_torques()
    SDU.plot_ee_pos_vel()
    trans_factor = 3
    csv_unproc_path = 'output/unprocessed_csv/unprocessed_{}'.format(now)
    csv_unproc_name = 'sim_data_{}.csv'.format(i)
    DFD.contact_info_to_csv(csv_unproc_name, csv_unproc_path,
                            trans_factor)  # Unprocessed data stored in csv file

    plt.figure()
    plt.axvline(x=kk * dt, linestyle='dashed', color='green')
    plt.plot(step[kk - k:], dframe[kk - k:, 2], color='#ba0001')
    plt.plot(steps, force_sensor, color='#125488')
    plt.legend(['Starting data recording', 'Reference force', 'Actual force'])
    plt.xlabel('$Time$')
    plt.ylabel('$Force \, [N]$')
    plt.grid()

DP.combine_trajectory_csv(csv_unproc_path, False)
csv_proc_path = 'output/processed_csv/processed_{}'.format(now)
csv_proc_name = 'sim_data_proc.csv'
csv_proc_name_norm = 'norm_data.csv'
DP.contact_info_processed_to_csv(csv_proc_name, csv_proc_path,
                                 csv_proc_name_norm)  # Creation of the dataset
DP.plot_data_comparison()
plt.show()
