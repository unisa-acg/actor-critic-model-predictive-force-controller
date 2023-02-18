import robosuite as suite
from robosuite.controllers import load_controller_config
import numpy as np
from datetime import datetime
import utils.quat_utils as quat_utils
import utils.utilities_ACMPFC as ACMPFC_utils
from utils.utilities_ACMPFC import modify_XML
import time
import matplotlib.pyplot as plt
import matplotlib
from utils.trajectory_generator import TrajectoryGenerator
from utils.trajectory_resampler import TrajectoryResampler
from utils.data_processing import DataProcessing
from utils.sim_data_utilities import SimulationDataUtilities
from utils.data_for_dataset import DataForDataset
import csv
from geometry_msgs.msg import PoseStamped, PointStamped, WrenchStamped, TwistStamped
import rospy
from std_msgs.msg import Bool
import torch.nn as nn
import torch
import os
matplotlib.use('QtCairo')

# Retrieve the distributions (mean,variance) from the ROS Param server
z_ee_dist, zdot_ee_dist, fz_ee_dist, error_f_dist, u_dist = ACMPFC_utils.get_distribution_parameters_lab(
)


### Prediction
def plot_traj():
    pos_z_norm = (pos_z - z_ee_dist[0]) / z_ee_dist[1]
    vel_z_norm = (vel_z - zdot_ee_dist[0]) / zdot_ee_dist[1]
    f_smoothed_norm = (f_smoothed - fz_ee_dist[0]) / fz_ee_dist[1]
    u_norm = (action_list - u_dist[0]) / u_dist[1]

    with torch.no_grad():
        model_approximator.eval()
        x = np.stack((pos_z_norm, vel_z_norm, f_smoothed_norm, u_norm), axis=-1)
        y_predicted = model_approximator.forward(torch.from_numpy(x)).cpu().numpy()
        f_pred = y_predicted[:, 2] * fz_ee_dist[1] + fz_ee_dist[0]
    f_pred = list(f_pred)
    force_temp = np.zeros(len(force_sensor))
    force_temp[-(len(f_pred)):] = f_pred

    t = np.arange(len(force_app), len(force_app) + len(dframe[:, 2]), 1)
    plt.figure()
    plt.axvline(x=len(df_approach) + 200, linestyle='dashed', color='yellow')
    plt.plot(t, dframe[:, 2], color='red')
    plt.plot(i_dis, force_sensor, color='#125488')
    plt.plot(i_dis, force_temp, color='black')
    plt.legend(
        ['Step when ACMPFC starts', 'Target force', 'Contact force', 'Force predicted'])
    plt.xlabel('steps')
    plt.ylabel('[N]')
    plt.title('Smoothed force from sensor')
    plt.grid()
    plt.xlim(left=3000)
    plt.ylim((0, 100))

    plt.figure()
    plt.plot(i_dis, action_list)
    plt.legend(['correction'])

    action_step = np.arange(0, len(u_list), 1) * 5 + start_ACMPFC_delay
    plt.figure()
    plt.plot(action_step, u_list)
    plt.legend(['u', 'u_limited'])

    plt.figure()
    plt.plot(i_dis, pos_z)
    plt.xlabel('steps')
    plt.ylabel('[m]')
    plt.legend(['pos_real'])


def neural_networks_instantiation(num_ensembles, Device):
    # Instantiate the three neural networks models
    actor = ACMPFC_utils.NeuralNetwork(num_inputs=4,
                                      num_outputs=2,
                                      num_hidden_layers=2,
                                      num_hidden_neurons=300,
                                      dropout_prob=0.1).to(Device)

    critic = ACMPFC_utils.NeuralNetwork_Critic(num_inputs=1,
                                              num_outputs=1,
                                              num_hidden_layers=2,
                                              num_hidden_neurons=128,
                                              dropout_prob=0.1).to(Device)

    model_approximator = ACMPFC_utils.NeuralNetwork(num_inputs=4,
                                                   num_outputs=3,
                                                   num_hidden_layers=3,
                                                   num_hidden_neurons=300,
                                                   dropout_prob=0).to(Device)

    return [actor, critic, model_approximator]


device = 'cpu'
[actor, critic, model_approximator] = neural_networks_instantiation(1, device)
model_approximator_loss = nn.MSELoss()
model_approximator_optim = torch.optim.Adam(model_approximator.parameters(),
                                            lr=1e-3,
                                            weight_decay=4e-5)

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

# Create an environment to visualize on-screen
env = suite.make(
    "Lift",
    robots=["Panda"],  # load a Panda robot
    gripper_types=None,  # use not default grippers per robot arm
    controller_configs=controller_config,  # the arm is controlled using OSC
    env_configuration="single-arm-opposed",  # single arm env
    has_renderer=False,  # on-screen rendering
    render_camera="frontview",  # visualize the "frontview" camera
    has_offscreen_renderer=False,  # no off-screen rendering
    control_freq=500,  # 500 hz control for applied actions
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
num_traj = 1  # Number of trajectories
ee_pos = env._eef_xpos  # End-effector position (point from which the
# approach trajectory starts)
table_pos = env.sim.model.body_pos[1]  # Table position (point in which the
# approach trajectory finishes)
gap = 0.062  # Gap added to table position to ensure minimal
# penetration
kp_pos = 150  # Gain of the position controller
kp_f = 0.1  # Gain of the force controller
old_time_step = 0.01  # Time step of the generated trajectory
new_time_step = 0.002  # Time step of the resampled trajectory (500Hz)
now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

## Class instantiation
TG = TrajectoryGenerator(old_time_step)
T = TrajectoryResampler()
DP = DataProcessing()

D_list = []
u_list = []
counter = 0

class RosNode:
    # Instantiate all the required publishers and subscribers 
    def __init__(self, name):
        rospy.init_node(name)
        rospy.loginfo("Starting MuJoCo")

        def myhook():
            print("[Sim] Simulator shutting down...")

        rospy.on_shutdown(myhook)

        self.publisher1 = rospy.Publisher('/franka_ee_pose', PoseStamped, queue_size=10)
        self.publisher2 = rospy.Publisher('/franka_ee_wrench',
                                          WrenchStamped,
                                          queue_size=10)
        self.publisher3 = rospy.Publisher('/franka_ee_velocity',
                                          TwistStamped,
                                          queue_size=10)
        self.publisher6 = rospy.Publisher('/error_f', WrenchStamped, queue_size=10)
        self.publisher7 = rospy.Publisher('/done', PointStamped, queue_size=10)

        self.u_sub = rospy.Subscriber("/u",
                                      PointStamped,
                                      callback=self.action_callback,
                                      queue_size=10,
                                      buff_size=2**20)
        self.received_sub = rospy.Subscriber("/done_received",
                                             Bool,
                                             callback=self.received_callback,
                                             queue_size=10,
                                             buff_size=2**20)
        self.action_received = True
        self.received = False

    def received_callback(self, received_msg):
        self.received = True

    def action_callback(self, u_msg):
        self.u = u_msg.point.x
        self.action_received = True

    def publish(self, msg_pose, msg_wrench, msg_vel, msg_error_f, msg_done):
        self.publisher1.publish(msg_pose)
        self.publisher2.publish(msg_wrench)
        self.publisher3.publish(msg_vel)
        self.publisher6.publish(msg_error_f)
        self.publisher7.publish(msg_done)


msg_pose = PoseStamped()
msg_vel = TwistStamped()
msg_wrench = WrenchStamped()
msg_error_f = WrenchStamped()
msg_done = PointStamped()

p_z_old = 0
print("Mujoco node is started")
mujoco_node = RosNode('mujoco')

start_ACMPFC_delay = 100  # 2600
action_list = []
u_old = 0
f_pred = []
vel_z = []
pos_z_csv = []
vel_z_csv = []
f_smoothed_csv = []
action_list_csv = []
error_f_csv = []

## Trajectories generation
for i in range(num_traj):
    # APPROACHING TRAJECTORY
    # Parameters definition
    waypoints = [(ee_pos[0], ee_pos[2]), (table_pos[0], table_pos[2] + gap + 0.05),
                 (table_pos[0], table_pos[2] + gap)]
    # waypoints = [(ee_pos[0], ee_pos[2]), (ee_pos[0], ee_pos[2] + 0.00001),
    #              (ee_pos[0], ee_pos[2])]
    traj_types = ['line', 'line']
    traj_params = [None, None]
    traj_timestamps = [0, 1, 6]
    force_reference_types = ['cnst', 'cnst']
    force_reference_parameters = [0, 0]

    # Trajectory generation
    [traj_app_x, traj_app_z,
     traj_app_f] = TG.traj_gen(waypoints, traj_types, traj_params, traj_timestamps,
                               force_reference_types, force_reference_parameters)
    TG.print_to_csv('traj_gen_{}.csv'.format(i), 'csv_folder/traj_gen_csv')

    time_vect = np.arange(traj_timestamps[0], traj_timestamps[-1], old_time_step)

    traj_matrix = T.read_traj_from_csv(
        'csv_folder/traj_gen_csv/traj_gen_{}.csv'.format(i))

    # Trajectory resampler
    T.interp_traj(traj_matrix, time_vect, new_time_step)
    T.traj_res_csv('traj_res_{}.csv'.format(i), 'csv_folder/traj_resampled_csv')
    df_approach = T.read_traj_from_csv(
        'csv_folder/traj_resampled_csv/traj_res_{}.csv'.format(i))

    #Main trajectories
    waypoints = [(table_pos[0], table_pos[1]), (-0.15, 0), (table_pos[0], table_pos[1]),
                 (-0.2, 0)]

    traj_types = ['line', 'line', 'line']
    traj_params = [None, None, None]

    traj_timestamps = [0, 5, 10, 20]
    force_reference_types = ['ramp', 'cnst', 'sine_curve']
    force_reference_parameters = [[25, 35], 35, [35, 15, 10, 15]]

    params_randomizer = {
        "operating_zone_points": [(-0.25, -0.25), (0.25, 0.25)]
    }  # first is y, second is x

    # Trajectory generation
    [x, y, f] = TG.traj_gen(waypoints, traj_types, traj_params, traj_timestamps,
                            force_reference_types, force_reference_parameters)
    TG.print_to_csv('traj_gen_{}.csv'.format(i), 'csv_folder/traj_gen_csv')

    # fig = TG.plot_traj(x, y, params_randomizer)  # Plot the generated trajectory
    # fig2 = TG.plot_force_ref(f, traj_timestamps)  # Plot the resampled trajectory

    time_vect = np.arange(traj_timestamps[0], traj_timestamps[-1], old_time_step)
    traj_matrix = T.read_traj_from_csv(
        'csv_folder/traj_gen_csv/traj_gen_{}.csv'.format(i))

    T.interp_traj(traj_matrix, time_vect, new_time_step)
    T.traj_res_csv('traj_res_{}.csv'.format(i), 'csv_folder/traj_resampled_csv')
    dframe = T.read_traj_from_csv(
        'csv_folder/traj_resampled_csv/traj_res_{}.csv'.format(i))

    traj_app_y = np.ones(len(df_approach)) * ee_pos[1]
    table_z = np.ones(len(dframe)) * (table_pos[2] + gap)
    force_app = np.ones(len(df_approach)) * np.array(dframe[:, 2], dtype=np.float64)[0]

    i_dis = []
    flag = 0
    force_sensor = []
    flag_dataset = 0
    flag_dataset_2 = 0
    pos_z = []
    ori_z = []
    steps = []
    f_spring = []
    f_smoothed = []
    pos_imp = []
    vel_imp = []
    corr = []
    vel_real = []
    vel_ctrl = []
    u_h_ctrl = []
    x_f_plot = []

    K_d = 350

    u_h_old = 0.8479
    u_h_vel = 0
    dt = 0.002

    smooth_ft_buffer = []  #deque(maxlen=20)
    smooth_ft_buffer.append(np.array([0, 0, 0, 0, 0, 0]))

    ####TOLTI ATTRITI
    env.sim.model.geom_solref[7][0] = 0.2
    env.sim.model.geom_solref[30][0] = 0.2
    env.sim.model.geom_friction[7] = [0.0, 0.005,
                                      0.0001]  
    env.sim.model.geom_friction[30] = [0.0, 0.005, 0.0001] 

    #traj building
    traj_approach = np.stack(
        (
            np.array(df_approach[:, 0], dtype=np.float64),
            traj_app_y,
            np.array(df_approach[:, 1], dtype=np.float64),
        ),
        axis=-1,
    )
    main_traj = np.stack(
        (
            np.array(dframe[:, 0], dtype=np.float64),
            np.array(dframe[:, 1], dtype=np.float64),
            table_z,
        ),
        axis=-1,
    )

    target_traj = np.concatenate((traj_approach, main_traj))

    force_ref = np.concatenate((force_app, np.array(dframe[:, 2], dtype=np.float64)))

    now_r = time.time()
    DFD = DataForDataset(len(target_traj))
    SDU = SimulationDataUtilities(env.horizon)

    ori_along_z_init = quat_utils.mat2euler(
        np.array(env.sim.data.site_xmat[env.sim.model.site_name2id(
            'gripper0_grip_site')].reshape([3, 3])))[2]

    #Trajectory execution
    for ii in range(len(target_traj)):
        controller = env.robots[0].controller
        SDU.contact_forces(env, 'table_collision', 'ee_sphere_collision')

        pos_along_z = env.sim.data.site_xpos[env.sim.model.site_name2id(
            'gripper0_grip_site')][2]
        vel_along_z = env.sim.data.site_xvelp[env.sim.model.site_name2id(
            'gripper0_grip_site')][2]

        ori_along_z = quat_utils.mat2euler(
            np.array(env.sim.data.site_xmat[env.sim.model.site_name2id(
                'gripper0_grip_site')].reshape([3, 3])))[2]

        smooth_ft_buffer.append(np.array(env.sim.data.sensordata))
        if len(smooth_ft_buffer) > 5:
            smooth_ft_buffer.pop(0)
        force_vals = np.mean(np.asarray(smooth_ft_buffer).copy(), 0)

        if ii == len(df_approach) + 1000 or flag_dataset == 1:
            flag_dataset = 1

        if ii == len(df_approach) + 100:
            flag = 1
            k = ii

        if flag == 1:
            f_d = force_ref[ii]
            with open('f_d_sine_45_7.csv', 'a') as f:
                write = csv.writer(f)
                write.writerow([f_d])

            f_e = env.sim.data.sensordata[2] 

            #Force controller
            x_f = FC.force_controller(f_d, f_e, dt)

            vel_real.append(vel_along_z)
            steps.append(ii)
            x_f_plot.append(target_traj[ii, 2] + x_f)

        if flag == 1:
            traj = np.array(
                [target_traj[ii, 0], target_traj[ii, 1], target_traj[ii, 2]])
            kp_pos = 500 
            kp_ori = 50
            kp_z = 500
            KP = np.array([kp_pos, kp_pos, kp_z, kp_ori, kp_ori, kp_ori])

        else:
            traj = np.array(target_traj[ii, :])
            kp_pos = 500  
            kp_ori = 50
            kp_z = 500
            KP = np.array([kp_pos, kp_pos, kp_z, kp_ori, kp_ori, kp_ori])

        #ALTRO
        pose = np.append(traj, np.array([0, -np.pi, 0]))
        action = np.append(KP, pose)

        force_sensor.append(env.sim.data.sensordata[2])
        i_dis.append(ii)
        ori_z.append(ori_along_z)

        SDU.get_info_robot(env, 'gripper0_grip_site')
        pos_along_z_old = pos_along_z

        # ---------------------------------------------------------------------------- #

        if flag == 1:
            msg_pose.pose.position.z = pos_along_z  # Pos z
            msg_vel.twist.linear.z = vel_along_z  # Vel z
            msg_wrench.wrench.force.z = env.sim.data.sensordata[2]  # Force z
            msg_error_f.wrench.force.z = env.sim.data.sensordata[2] - force_ref[ii]
            msg_done.point.x = 0

        only_force = False
        save_data = False

        if flag == 1:
            mujoco_node.action_received = False
            mujoco_node.publish(msg_pose, msg_wrench, msg_vel, msg_error_f, msg_done)

            if only_force:
                mujoco_node.u = 0
            else:
                while mujoco_node.action_received is False:
                    counter += 1
            u_list.append(mujoco_node.u)

            if mujoco_node.action_received:
                delta = 1
                u = np.clip(mujoco_node.u, u_old - delta,
                            u_old + delta)  # mujoco_node.u
                u_old = u
                if only_force == True:
                    pass
                else:
                    action[8] = target_traj[
                        ii,
                        2] + u 
                    action_for_five = u

        ### STORE DATA TO CSV
        if flag == 1:
            pos_z_csv.append(pos_along_z)
            vel_z_csv.append(vel_along_z)
            f_smoothed_csv.append(env.sim.data.sensordata[2])
            action_list_csv.append(x_f)
            error_f_csv.append(env.sim.data.sensordata[2] - f_d)

        ### STORE DATA TO PLOT
        pos_z.append(pos_along_z)
        vel_z.append(vel_along_z)
        f_smoothed.append(env.sim.data.sensordata[2])
        if flag == 0:
            x_f = 0
            u = 0
        if only_force:
            action_list.append(x_f)
        else:
            action_list.append(u)
        # ---------------------------------------------------------------------------- #
        if flag == 1:
            if (env.sim.data.sensordata[2] < 0 or env.sim.data.sensordata[2] > 70):
                for _ in range(50):
                    msg_pose.pose.position.z = pos_along_z  # Pos z
                    msg_vel.twist.linear.z = vel_along_z  # Vel z
                    msg_wrench.wrench.force.z = env.sim.data.sensordata[2]  # Force z
                    msg_error_f.wrench.force.z = env.sim.data.sensordata[2] - force_ref[ii]
                    msg_done.point.x = 0
                    mujoco_node.publish(msg_pose, msg_wrench, msg_vel, msg_error_f, msg_done)
                print('Threshold force')
                break

        env.step(action)
        # env.render()

    print('Simulation finished, sending signal for replay training')
    msg_pose.pose.position.z = pos_along_z  # Pos z
    msg_vel.twist.linear.z = vel_along_z  # Vel z
    msg_wrench.wrench.force.z = env.sim.data.sensordata[2]  # Force z
    msg_error_f.wrench.force.z = env.sim.data.sensordata[2] - force_ref[ii]
    msg_done.point.x = 1
    time.sleep(2)
    mujoco_node.publish(msg_pose, msg_wrench, msg_vel, msg_error_f, msg_done)

    env.reset_from_xml_string(xml_string)

    def write_data_to_csv(path_data_saves):

        mean_pos_z = np.mean(pos_z_csv)
        mean_vel_z = np.mean(vel_z_csv)
        mean_f_smooth = np.mean(f_smoothed_csv)
        mean_u = np.mean(action_list_csv)
        mean_error_f = np.mean(error_f_csv)
        std_dev_pos_z = np.std(pos_z_csv)
        std_dev_vel_z = np.std(vel_z_csv)
        std_dev_f_smooth = np.std(f_smoothed_csv)
        std_dev_u = np.std(action_list_csv)
        std_dev_error_f = np.std(error_f_csv) 

        means = [mean_pos_z, mean_vel_z, mean_f_smooth, mean_u, mean_error_f]
        std_devs = [
            std_dev_pos_z, std_dev_vel_z, std_dev_f_smooth, std_dev_u, std_dev_error_f
        ]

        pos_z_norm = (pos_z_csv - mean_pos_z) / std_dev_pos_z
        vel_z_norm = (vel_z_csv - mean_vel_z) / std_dev_vel_z
        f_smoothed_norm = (f_smoothed_csv - mean_f_smooth) / std_dev_f_smooth
        u_norm = (action_list_csv - mean_u) / std_dev_u
        error_norm = (error_f_csv - mean_error_f) / std_dev_error_f

        data_matrix = np.vstack(
            (pos_z_norm, vel_z_norm, f_smoothed_norm, u_norm, error_norm)).transpose()

        with open(os.path.join(path_data_saves, 'data_sine_5_50_5_noD.csv'), 'w') as f:
            write = csv.writer(f)
            header = ['z_ee_t', 'zdot_ee_t', 'f_z_ee_t', 'u_t', 'e_f']
            write.writerow(header)
            write.writerows(data_matrix)
        with open(os.path.join(path_data_saves, 'norm_data_sine_5_50_5_noD.csv'),
                  'w') as f:
            write = csv.writer(f)
            header = ['z_ee_t', 'zdot_ee_t', 'f_z_ee_t', 'u_t', 'e_f']
            write.writerow(header)
            write.writerow(means)
            write.writerow(std_devs)

        return pos_z_norm, vel_z_norm, f_smoothed_norm, u_norm, mean_f_smooth, std_dev_f_smooth

    if save_data == True:
        PATH = 'csv_data_franka'
        write_data_to_csv(PATH)
    plot_traj()

MSE = (np.power(error_f_csv, 2)).mean()

print('MSE tracking:', MSE)

plt.show()
