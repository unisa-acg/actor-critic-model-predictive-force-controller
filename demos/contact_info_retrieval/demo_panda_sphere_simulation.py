import os
import time
import numpy as np
from mujoco_panda import PandaArm
from mujoco_panda.utils.viewer_utils import render_frame
from mujoco_panda.utils.debug_utils import ParallelPythonCmd
from mujoco_panda.controllers.torque_based_controllers import (
    OSHybridForceMotionController, )
import matplotlib.pyplot as plt
import mujoco_validation.contact_forces_validation as validate

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
    panda_arm = PandaArm(
        model_path=MODEL_PATH + "panda_block_table.xml",
        render=True,
        compensate_gravity=False,
        smooth_ft_sensor=True,
    )

    force_reference_z = 50  # reference force for force control

    panda_arm.set_neutral_pose()
    panda_arm.step()

    # create controller instance with default controller gains
    hybrid_force_controller = OSHybridForceMotionController(panda_arm,
                                                            config=ctrl_config)

    # --- define trajectory in position -----
    curr_pos, curr_ori = panda_arm.ee_pose()
    goal_pos = curr_pos.copy()
    goal_pos[2] = 0.5
    target_traj = np.linspace(curr_pos, goal_pos, 200)
    target_ori = curr_ori

    # activate controller
    hybrid_force_controller.set_active(True)

    now_r = time.time()
    i = 0

    # Instantiate the class to calculate the contact forces with explicit and built in method
    contact_forces_validation = validate.MujocoContactValidation(
        panda_arm.sim, target_traj.shape[0])

    for waypoint in target_traj:

        # Get current robot end-effector pose
        robot_pos, robot_ori = panda_arm.ee_pose()

        elapsed_r = time.time() - now_r

        # Render controller target and current ee pose using frames
        render_frame(panda_arm.viewer, robot_pos, robot_ori)
        render_frame(panda_arm.viewer, waypoint, target_ori, alpha=0.2)

        hybrid_force_controller.change_ft_dir(
            [1, 1, 1, 1, 1, 1])  # start force control along Z axis
        hybrid_force_controller.set_goal(
            waypoint,
            target_ori,
            goal_force=[0, 0, -force_reference_z],
            goal_torque=[0, 0, 0],
        )  # target force in cartesian frame

        # --- If needed uncomment, it renders the visualization at a slower time rate ---
        # if elapsed_r >= 0.1:
        #     i += 1  # change target less frequently compared to render rate
        #     #print ("smoothed FT reading: ", p.get_ft_reading(pr=True))
        #     now_r = time.time()
        # -------------------------------------------------------------------------------

        panda_arm.render()  # render the visualisation

        contact_forces_validation.contact_forces(panda_arm.sim)

        # Store results in csv file
        contact_forces_validation.contact_forces_to_csv(panda_arm.sim)

    hybrid_force_controller.set_active(False)

    contact_forces_validation.plot_contact_forces()

    plt.show()

    input("Trajectory completed. Hit Enter to deactivate controller")
    hybrid_force_controller.set_active(False)
    hybrid_force_controller.stop_controller_cleanly()
