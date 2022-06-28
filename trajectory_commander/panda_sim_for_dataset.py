import robosuite as suite
from robosuite.controllers import load_controller_config
import numpy as np
import time
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib

import trajectories.randomizer as R
from trajectories.traj_gen import TrajectoryGenerator
from trajectories.traj_resampler import TrajectoryResampler

from data_handling.data_for_dataset import DataForDataset
from data_handling.data_processing import DataProcessing
from data_handling.sim_data_utilities import SimulationDataUtilities

matplotlib.use('QtCairo')

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

    #print(ET.tostring(root, encoding='utf8').decode('utf8'))
    xml_string = ET.tostring(root, encoding='utf8').decode('utf8')
    tree.write('sim_model_mod.xml')

    return xml_string


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
controller_config['impedance_mode'] = 'fixed'

# create an environment to visualize on-screen
env = suite.make(
    "Lift",
    robots=["Panda"],  # load a Sawyer robot and a Panda robot
    gripper_types=None,  # use default grippers per robot arm #"WipingGripper"
    controller_configs=controller_config,  # each arm is controlled using OSC
    env_configuration=
    "single-arm-opposed",  # (two-arm envs only) arms face each other
    has_renderer=True,  # on-screen rendering
    render_camera="frontview",  # visualize the "frontview" camera
    has_offscreen_renderer=False,  # no off-screen rendering
    control_freq=500,  # 20 hz control for applied actions
    horizon=500000,  # each episode terminates after 200 steps
    use_object_obs=False,  # no observations needed
    use_camera_obs=False,  # no observations needed
)

# env.sim.model.opt.solver = 0
# env.sim.model.opt.cone = 1
# env.sim.model.opt.noslip_iterations = 50000
# env.sim.model.opt.impratio = 2000

suite.utils.mjcf_utils.save_sim_model(env.sim, 'sim_model.xml')
xml_string = modify_XML()
env.reset_from_xml_string(xml_string)

# env.sim.model.opt.solver = 0
# env.sim.model.opt.cone = 1
# env.sim.model.opt.noslip_iterations = 50000
# env.sim.model.opt.impratio = 2000

#env.robots[0].inizialization_noise = None
smooth_ft_buffer = []  #deque(maxlen=20)
smooth_ft_buffer.append(np.array([0, 0, 0, 0, 0, 0]))
####

TG = TrajectoryGenerator()
T = TrajectoryResampler()
SDU = SimulationDataUtilities(env.horizon)
DP = DataProcessing()

num_traj = 1
ee_pos = env._eef_xpos
table_pos = env.sim.model.body_pos[1]
gap = 0.072  
kp_pos = 150
kp_f = 0.1
old_time_step = 0.1
new_time_step = 0.002

for i in range(num_traj):
    #Approaching trajectories
    waypoints = [(ee_pos[0], ee_pos[2]), (table_pos[0], table_pos[2] + gap + 0.05),
                (table_pos[0], table_pos[2] + gap)]
    traj_types = ['line', 'line']
    traj_params = [None, None]
    traj_timestamps = [0, 1, 5]
    force_reference_types = ['cnst', 'cnst']
    force_reference_parameters = [0, 0]
    [traj_app_x, traj_app_z,
    traj_app_f] = TG.traj_gen(waypoints, traj_types, traj_params, traj_timestamps,
                            force_reference_types, force_reference_parameters)
    TG.print_to_csv('csv_folder/traj_gen_{}.csv'.format(i))

    time_vect = np.arange(traj_timestamps[0], traj_timestamps[-1], old_time_step)

    traj_matrix = T.read_traj_from_csv('csv_folder/traj_gen_{}.csv'.format(i))

    T.interp_traj(traj_matrix, time_vect, new_time_step)
    T.traj_res_csv('csv_folder/traj_res_{}.csv'.format(i))
    df_approach = T.read_traj_from_csv('csv_folder/traj_res_{}.csv'.format(i))

    #Main trajectory
    steps_required = 1e6
    while steps_required > 1e5:
        params_randomizer = {
            'starting_point': (0, 0),
            "operating_zone_points": [(-0.25, -0.25),
                                    (0.22, 0.22)],  # il primo è y il secondo x
            'max_n_subtraj': 3,
            'max_vel': 0.016,
            'max_radius': 0.1,
            'min_radius': 0.01,
            'max_ampl': 0.1,
            'max_freq': 10,
            'min_f_ref': 10,
            'max_f_ref': 80
        }
        waypoints, traj_timestamps, traj_types, traj_params, force_reference_types, force_reference_parameters = R.traj_randomizer(
            params_randomizer)
        steps_required = (traj_timestamps[-1] + 6) / 0.002

    print('Steps required: ', steps_required)

    [x, y, f] = TG.traj_gen(waypoints, traj_types, traj_params, traj_timestamps,
                            force_reference_types, force_reference_parameters)
    TG.print_to_csv('csv_folder/traj_gen_{}.csv'.format(i))

    fig = TG.plot_traj(x, y, params_randomizer)

    fig2 = TG.plot_force_ref(f, traj_timestamps)

    time_vect = np.arange(traj_timestamps[0], traj_timestamps[-1], old_time_step)

    traj_matrix = T.read_traj_from_csv('csv_folder/traj_gen_{}.csv'.format(i))

    T.interp_traj(traj_matrix, time_vect, new_time_step)
    T.traj_res_csv('csv_folder/traj_res_{}.csv'.format(i))
    dframe = T.read_traj_from_csv('csv_folder/traj_res_{}.csv'.format(i))

    traj_app_y = np.ones(len(df_approach)) * ee_pos[1]
    table_z = np.ones(len(dframe)) * (table_pos[2] + gap)
    force_app = np.ones(len(df_approach)) * np.array(dframe[:,2], dtype=np.float64)[0]       

    i_dis = []
    flag = 0
    flag_dataset = 0
    force_sensor = []
    smooth_ft_buffer = []  #deque(maxlen=20)
    smooth_ft_buffer.append(np.array([0, 0, 0, 0, 0, 0]))

    ####TOLTI ATTRITI
    env.sim.model.geom_condim[:] = 6
    # env.sim.model.geom_solref[7][0] = 0.01
    # env.sim.model.geom_solref[30][0] = 0.01
    # env.sim.model.geom_solimp[7] = [0.99, 0.99, 0.001, 0.5, 1]
    # env.sim.model.geom_solimp[30] = [0.99, 0.99, 0.001, 0.5, 1]
    env.sim.model.geom_friction[7] = [0.03, 0.005, 0.0001
                                    ]  #0.2 #default[1.0, 0.005, 0.0001]
    env.sim.model.geom_friction[30] = [0.03, 0.005, 0.0001]  #0.2

    #traj building
    traj_approach = np.stack(
        (
            np.array(df_approach[:,0], dtype=np.float64),
            traj_app_y,
            np.array(df_approach[:,1], dtype=np.float64),
        ),
        axis=-1,
    )
    main_traj = np.stack(
        (
            np.array(dframe[:,0], dtype=np.float64),
            np.array(dframe[:,1], dtype=np.float64),
            table_z,
        ),
        axis=-1,
    )

    target_traj = np.concatenate((traj_approach, main_traj))

    force_ref = np.concatenate(
        (force_app, np.array(dframe[:,2], dtype=np.float64)))

    now_r = time.time()
    DFD = DataForDataset(len(target_traj))

    #Trajectory execution
    for ii in range(len(target_traj)):
        controller = env.robots[0].controller
        SDU.contact_forces(env, 'table_collision', 'ee_sphere_collision')

        smooth_ft_buffer.append(np.array(env.sim.data.sensordata))
        if len(smooth_ft_buffer) > 5:
            smooth_ft_buffer.pop(0)
        force_vals = np.mean(np.asarray(smooth_ft_buffer).copy(), 0)

        #force controller activation
        if env.check_contact('ee_sphere_collision',
                            'table_collision') or flag == 1:
            # F.force_control_activation(force_vals, force_ref)
            controller.activate_pid_z = True
            controller.fs = force_vals[2]

            controller.f_target = force_ref[ii]
            controller.kp_pid = 0.08  #0.3
            controller.kd_pid = 350
            controller.ki_pid = 0.001  #0.001
            flag = 1

        if ii == len(df_approach) + 600 or flag_dataset == 1:
            DFD.get_info_robot(env, 'robot0_right_hand')
            flag_dataset = 1

        action = np.append(target_traj[ii, :], np.array([0, -np.pi, 0]))
        env.step(action)

        force_sensor.append(force_vals[2])
        i_dis.append(ii)

        SDU.get_info_robot(env, 'robot0_right_hand')

        # t = time.time() - now_r
        # if ii == 1500:
        #     print(1/(t/1500))

        add_markers_op_zone(env, params_randomizer, table_pos[2])

        #env.render()

    env.reset_from_xml_string(xml_string)
    #F.reset_force_controller()
    controller.integral = 0
    controller.activate_pid_z = False

    SDU.plot_contact_forces()
    SDU.plot_torques()
    SDU.plot_ee_pos_vel()
    DFD.contact_info_to_csv("csv_folder/sim_data_{}.csv".format(i))
    DP.read_data_from_csv("csv_folder/sim_data_{}.csv".format(i))
    DP.contact_info_processed_to_csv("csv_folder/sim_data_proc_{}.csv".format(i))
    DP.plot_data_comparison()

    plt.figure()
    plt.plot(i_dis, force_sensor)
    

plt.show()