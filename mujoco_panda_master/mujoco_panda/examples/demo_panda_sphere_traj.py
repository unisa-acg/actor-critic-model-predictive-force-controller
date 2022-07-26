import csv
import logging
import os
import time

import matplotlib.pyplot as plt
import mujoco_py
import numpy as np
from mujoco_panda import PandaArm
from mujoco_panda.controllers.torque_based_controllers import (
    OSHybridForceMotionController,
)
from mujoco_panda.utils.debug_utils import ParallelPythonCmd
from mujoco_panda.utils.viewer_utils import render_frame
from simple_exp import contact_forces_validation as validate
from Trajectories.traj_utilities.traj_utils import traj_utilities

# Model path
MODEL_PATH = os.environ["MJ_PANDA_PATH"] + "/mujoco_panda/models/"

# controller parameters
KP_P = np.array([7000.0, 7000.0, 7000.0])
KP_O = np.array([3000.0, 3000.0, 3000.0])
ctrl_config = {
    "kp_p": KP_P,
    "kd_p": 2.0 * np.sqrt(KP_P),
    "kp_o": KP_O,  # 10gains for orientation
    "kd_o": [1.0, 1.0, 1.0],  # gains for orientation
    "kp_f": [1.0, 1.0, 1.0],  # gains for force
    "kd_f": [0.0, 0.0, 0.0],  # 25gains for force
    "kp_t": [1.0, 1.0, 1.0],  # gains for torque
    "kd_t": [0.0, 0.0, 0.0],  # gains for torque
    "alpha": 3.14 * 0,
    "use_null_space_control": True,
    "ft_dir": [0, 0, 0, 0, 0, 0],
    # newton meter
    "null_kp": [5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0],
    "null_kd": 0,
    "null_ctrl_wt": 2.5,
    "use_orientation_ctrl": True,
    "linear_error_thr": 0.025,
    "angular_error_thr": 0.01,
}

if __name__ == "__main__":

    # Instantiate the panda model simulator
    p = PandaArm(
        model_path=MODEL_PATH + "panda_block_table.xml",
        render=True,
        compensate_gravity=False,
        smooth_ft_sensor=True,
    )

    force_vect = np.array([50])
    tu = traj_utilities()
    [df, traj_count] = tu.read_traj_from_csv(csv_file_path="dataset.csv")

    for force_index in range(force_vect.shape[0]):

        p.set_neutral_pose()
        p.step()

        # create controller instance with default controller gains
        ctrl = OSHybridForceMotionController(p, config=ctrl_config)

        # --- define trajectory in position -----
        curr_ee, curr_ori = p.ee_pose()
        goal_pos_1 = curr_ee.copy()
        goal_pos_1[2] = 0.5
        goal_pos_1[0] += 0
        goal_pos_1[1] -= 0
        target_traj_old = np.linspace(curr_ee, goal_pos_1, 200)
        target_traj_1 = np.stack(
            (
                np.array(df["x0"], dtype=np.float64),
                np.array(df["y0"], dtype=np.float64),
                np.array(df["z0"], dtype=np.float64),
            ),
            axis=-1,
        )
        z_target = curr_ee[2]
        target_traj = target_traj_1
        target_ori = curr_ori

        ctrl.set_active(
            True
        )  # activate controller (simulation step and controller thread now running)

        now_r = time.time()
        i = 0
        count = 0

        # Instantiate the class to calculate the contact forces with explicit and built in method
        Validate = validate.MujocoContactValidation(p.sim, target_traj.shape[0])

        while i < target_traj.shape[0]:

            # Get current robot end-effector pose
            robot_pos, robot_ori = p.ee_pose()

            elapsed_r = time.time() - now_r

            # Render controller target and current ee pose using frames
            render_frame(p.viewer, robot_pos, robot_ori)
            render_frame(p.viewer, target_traj[i, :], target_ori, alpha=0.2)

            if i == 9000:  # activate force control when the robot is near the table
                ctrl.change_ft_dir(
                    [1, 1, 1, 1, 1, 1]
                )  # start force control along Z axis
                ctrl.set_goal(
                    target_traj[i, :],
                    target_ori,
                    goal_force=[0, 0, -force_vect[force_index]],
                    goal_torque=[0, 0, 0],
                )  # target force in cartesian frame
            else:
                ctrl.set_goal(target_traj[i, :], target_ori)

            # --- If needed uncomment, it renders the visualization at a slower time rate ---
            # if elapsed_r >= 0.1:
            #     i += 1 # change target less frequently compared to render rate
            #     #print ("smoothed FT reading: ", p.get_ft_reading(pr=True))
            #     now_r = time.time()
            # -------------------------------------------------------------------------------

            p.render()  # render the visualisation

            Validate.contact_forces(p.sim)

            # Store results in csv file
            Validate.contact_forces_to_csv(p.sim)

            i += 1

        ctrl.set_active(False)
        print(p.ee_pose())

    Validate.plot_contact_forces()

    plt.show()

    input("Trajectory complete. Hit Enter to deactivate controller")
    ctrl.set_active(False)
    ctrl.stop_controller_cleanly()
