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

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Comic Sans MS"

logging.getLogger("matplotlib.font_manager").disabled = True


def exec_func(cmd):
    if cmd == "":
        return None
    a = eval(cmd)
    print(cmd)
    print(a)
    if a is not None:
        return str(a)


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

    p = PandaArm(
        model_path=MODEL_PATH + "panda_block_table.xml",
        render=True,
        compensate_gravity=False,
        smooth_ft_sensor=True,
    )
    # force_vect = np.linspace(25, 100, num=10)
    force_vect = np.array([50])
    xdata = []
    ydata = []
    penetration_vect = []
    tau_contact = 0.01
    vect_step = []
    k_ratio_vect = []
    fz_vect = []
    eq_force = 0
    eq_pen = 0

    for force_index in range(force_vect.shape[0]):

        # cmd = ParallelPythonCmd(exec_func) # m 1 -1.2*10^-5
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
        target_traj_1 = np.linspace(curr_ee, goal_pos_1, 100)
        z_target = curr_ee[2]
        target_traj = target_traj_1
        target_ori = curr_ori

        ctrl.set_active(
            True
        )  # activate controller (simulation step and controller thread now running)

        now_r = time.time()
        i = 0
        count = 0

        p.model.geom_solref[17, 0] = tau_contact
        # print(p.model.geom_solref)

        while i < target_traj.shape[0]:
            # get current robot end-effector pose
            robot_pos, robot_ori = p.ee_pose()

            elapsed_r = time.time() - now_r

            # render controller target and current ee pose using frames
            render_frame(p.viewer, robot_pos, robot_ori)
            render_frame(p.viewer, target_traj[i, :], target_ori, alpha=0.2)

            if i == 10:  # activate force control when the robot is near the table
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

            if elapsed_r >= 0.1:
                i += 1  # change target less frequently compared to render rate
                # print ("smoothed FT reading: ", p.get_ft_reading(pr=True))
                now_r = time.time()

            p.render()  # render the visualisation

            sim = p.sim
            contact = sim.data.contact[0]
            pen = contact.dist
            penetration_vect.append(pen)
            vect_step.append(i)
            ft_reading = p.get_ft_reading(pr=True)
            forces_reading = ft_reading[0]
            fz_sensor = forces_reading[2]
            fz_vect.append(fz_sensor)

        print("smoothed FT reading: ", p.get_ft_reading(pr=True))
        ctrl.set_active(False)
        print(force_index)

        sim = p.sim
        contact = sim.data.contact[0]
        eq_pen = contact.dist
        eq_force = fz_sensor
        xdata.append(eq_pen)
        ydata.append(eq_force)
        tau_verifica = contact.solref[0]

        k_eq = force_vect[force_index] / eq_pen
        k = 1 / (tau_contact**2)
        k_ratio = k_eq / k
        k_ratio_vect.append(k_ratio)

    print("K tau", k)
    print("Equivalent stiffness:", k_eq)
    print("Ratio", k_ratio_vect)
    input("Press enter to plot...")
    plt.figure()
    plt.rcParams["text.usetex"] = True
    plt.title(r"$\tau = 0.01$")
    plt.ylabel(r"Force $[N]$")
    plt.xlabel(r"Penetration $[m]$")
    plt.grid()
    plt.plot(np.absolute(xdata), ydata, linewidth=2)
    plt.show()

    # reversexdata = xdata[::-1]

    input("Trajectory complete. Hit Enter to deactivate controller")
    ctrl.set_active(False)
    ctrl.stop_controller_cleanly()
