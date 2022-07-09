import dataset_generation.src.utilities.quat_utils as quat_utils
import matplotlib.pyplot as plt
import mujoco_py
import numpy as np


class SimulationDataUtilities:
    """
    Class that contains method to retrieve info happening during the MuJoCo simulation

    Methods
    -------
    * __init__(sim,steps): instantiate the variables need for the other methods

    * get_info_robot(env, bodies_name): it extracts the useful data from the simulation

    * contact_forces(env,geom1_name,geom2_name): retrieve contact forces data from simulation between the two specified geometries. In addition also penetration and velocity of penetration are stored.

    * plot_contact_forces(): plot the forces directions and the total magnitude, as well as the penetration data, happened between the pair of bodies specified in ``contact_forces()``.

    * plot_torques(): plot the torques of the robot frames.

    * plot_ee_pos_vel(): plot the position and velocity of the robot end-effector.
    """

    def __init__(self, steps):
        self.ncalls = 0
        self.forces_vect = np.zeros((6, steps), dtype=np.float64, order="C")
        self.dist = np.zeros((1, steps), dtype=np.float64, order="C")
        self.vel = np.zeros((1, steps), dtype=np.float64, order="C")

        self.ee_pos = np.zeros((steps, 3), dtype=np.float64, order="C")
        self.ee_ori = np.zeros((steps, 3), dtype=np.float64, order="C")
        self.ee_quat = np.zeros((steps, 4), dtype=np.float64, order="C")
        self.ee_vel = np.zeros((steps, 3), dtype=np.float64, order="C")
        self.ee_omg = np.zeros((steps, 3), dtype=np.float64, order="C")
        self.ee_force = np.zeros((steps, 3), dtype=np.float64, order="C")
        self.ee_torque = np.zeros((steps, 3), dtype=np.float64, order="C")
        self.sensor_data = np.zeros((steps, 6), dtype=np.float64, order="C")
        self.actuated_joints = np.zeros((steps, 7), dtype=np.float64, order="C")

        self.ncalls_get_info_robot = 0

    def get_info_robot(self, env, bodies_name):
        """It extracts the useful data from the simulation

        Args:
            env: MuJoCo environment from which allows access to sim (mjSim -> MuJoCo simulation instance at current timestep)
            bodies_name: name of the bodies whose contact data need to be obtained 
        """
        self.ee_force[self.ncalls_get_info_robot, :] = env.robots[0].ee_force
        self.ee_torque[self.ncalls_get_info_robot, :] = env.robots[0].ee_torque
        self.sensor_data[
            self.
            ncalls_get_info_robot, :] = env.sim.data.sensordata  #check that they are equal to ee_force and torque
        #ee_pos = env._get_observations()['robot0_eef_pos']
        #ee_pos = env.sim.data.get_site_xpos('gripper0_grip_site')
        #body = 'robot0_right_hand'  #'ee_sphere'
        self.ee_pos[self.ncalls_get_info_robot, :] = env.sim.data.get_body_xpos(
            bodies_name)
        #ee_quat = env._get_observations()['robot0_eef_quat']
        self.ee_quat[self.ncalls_get_info_robot, :] = env.sim.data.get_body_xquat(
            bodies_name)
        #ee_velp = env.sim.data.get_site_xvelp('gripper0_grip_site')
        self.ee_vel[self.ncalls_get_info_robot, :] = env.sim.data.get_body_xvelp(
            bodies_name)
        #ee_velp2 = env.sim.data.get_site_xvelp('gripper0_ft_frame')
        #ee_velr = env.sim.data.get_site_xvelp('gripper0_grip_site')
        self.ee_omg[self.ncalls_get_info_robot, :] = env.sim.data.get_body_xvelr(
            bodies_name)
        #ee_ori = quat2euler(ee_quat)
        #ee_ori = ee_quat
        self.ee_ori[self.ncalls_get_info_robot, :] = quat_utils.quat2euler(
            env.sim.data.get_body_xquat(bodies_name))

        self.actuated_joints[self.ncalls_get_info_robot, :] = env.sim.data.ctrl
        joint_indexes = env.robots[0].joint_indexes
        grip_site = 'gripper0_grip_site'

        J_pos = np.array(
            env.sim.data.get_site_jacp(grip_site).reshape((3, -1))[:, joint_indexes])
        J_ori = np.array(
            env.sim.data.get_site_jacr(grip_site).reshape((3, -1))[:, joint_indexes])
        self.J_full = np.array(np.vstack([J_pos, J_ori]))
        self.mjmodel = env.sim.model

        self.ncalls_get_info_robot += 1

    def contact_forces(self, env, geom1_name, geom2_name):
        """Substitute function to the heavy contact_forces(), which can slow down simulation times. It implements the built in methods to retrieve the forces even in presence of friction. Stores them in the class in order to plot them.

        Args:
            env: MuJoCo environment from which allows access to sim (mjSim -> MuJoCo simulation instance at current timestep)
            geom1_name, geom2_name: name of the geometries whose mutual contact you want to check
        """
        sim = env.sim
        env.check_contact("sphere", "table_collision")
        force_z = 0

        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            if (sim.model.geom_id2name(contact.geom1) == geom1_name or
                    sim.model.geom_id2name(contact.geom1) == geom2_name) and (
                        sim.model.geom_id2name(contact.geom2) == geom1_name or
                        sim.model.geom_id2name(contact.geom2) == geom2_name):
                forces_vect = np.zeros(6, dtype=np.float64)
                mujoco_py.functions.mj_contactForce(sim.model, sim.data, i, forces_vect)
                self.forces_vect[:, self.ncalls] = forces_vect
                efc_address = sim.data.contact[i].efc_address
                dist = sim.data.active_contacts_efc_pos[efc_address]
                vel = sim.data.efc_vel[efc_address]
                self.dist[:, self.ncalls] = dist
                self.vel[:, self.ncalls] = vel
        self.ncalls += 1

    def plot_contact_forces(self):
        """plot the forces directions and the total magnitude, as well as the penetration data, happened between the pair of bodies specified in ``contact_forces()``. Must be called AFTER ``contact_forces()`` method

        Returns:
            fig
        """
        fx = self.forces_vect[2, 0:self.ncalls]
        fy = self.forces_vect[1, 0:self.ncalls]
        fz = self.forces_vect[0, 0:self.ncalls]
        fxyz_sq = np.power(fx, 2) + np.power(fy, 2) + np.power(fz, 2)
        f_mag = np.power(fxyz_sq, 0.5)

        tx = self.forces_vect[5, 0:self.ncalls]
        ty = self.forces_vect[4, 0:self.ncalls]
        tz = self.forces_vect[3, 0:self.ncalls]

        penetration = self.dist[0, 0:self.ncalls]
        velocity = self.vel[0, 0:self.ncalls]

        step_vect = np.arange(0, self.ncalls, 1)
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(step_vect, fx)
        axs[0, 0].set_title('$F_x$')
        axs[0, 0].grid()
        axs[0, 1].plot(step_vect, fy, 'tab:orange')
        axs[0, 1].set_title('$F_y$')
        axs[0, 1].grid()
        axs[1, 0].plot(step_vect, fz, 'tab:green')
        axs[1, 0].set_title('$F_z$')
        axs[1, 0].grid()
        axs[1, 1].plot(step_vect, f_mag, 'tab:red')
        axs[1, 1].set_title('$F_{tot}$')
        axs[1, 1].grid()

        fig, axs = plt.subplots(1, 3)
        axs[0].plot(step_vect, tx)
        axs[0].set_title('$T_x$')
        axs[0].grid()
        axs[1].plot(step_vect, ty, 'tab:orange')
        axs[1].set_title('$T_y$')
        axs[1].grid()
        axs[2].plot(step_vect, tz, 'tab:green')
        axs[2].set_title('$T_z$')
        axs[2].grid()

        fig, axs = plt.subplots(1, 2)
        axs[0].plot(step_vect, penetration)
        axs[0].set_title('$Penetration$')
        axs[0].grid()
        axs[1].plot(step_vect, velocity, 'tab:orange')
        axs[1].set_title('$Velocity$')
        axs[1].grid()

        for ax in axs.flat:
            ax.set(xlabel='Steps', ylabel='$[N]$')

        return [axs, fig]

    def plot_torques(self):
        """plot the torques of the robot frames. Must be called AFTER ``get_info_robot()`` method

        Returns:
            fig
        """
        step_vect = np.arange(0, self.ncalls_get_info_robot, 1)
        n_graph = 0

        fig, axs = plt.subplots(2, 3)
        for j in range(2):
            for k in range(3):
                axs[j,
                    k].plot(step_vect,
                            self.actuated_joints[0:self.ncalls_get_info_robot, n_graph])
                axs[j, k].set_title('$torque_{}$'.format(n_graph))
                axs[j, k].grid()
                n_graph += 1

        for ax in axs.flat:
            ax.set(xlabel='Steps', ylabel='$[N]$')

        return [axs, fig]

    def plot_ee_pos_vel(self):
        """plot the position and velocity of the robot end-effector. Must be called AFTER ``get_info_robot()`` method
        Returns:
            fig
        """
        step_vect = np.arange(0, self.ncalls_get_info_robot, 1)

        fig, axs = plt.subplots(1, 3)
        axs[0].plot(step_vect, self.ee_pos[0:self.ncalls_get_info_robot, 0])
        axs[0].set_title('$Ee \; pos_x$')
        axs[0].grid()
        axs[1].plot(step_vect, self.ee_pos[0:self.ncalls_get_info_robot, 1],
                    'tab:orange')
        axs[1].set_title('$Ee \; pos_y$')
        axs[1].grid()
        axs[2].plot(self.ee_pos[0:self.ncalls_get_info_robot, 0],
                    self.ee_pos[0:self.ncalls_get_info_robot, 1], 'tab:red')
        axs[2].set_title('$Ee \; pos_{xy}$')
        axs[2].grid()

        fig, axs = plt.subplots(1, 2)
        axs[0].plot(step_vect, self.ee_vel[0:self.ncalls_get_info_robot, 0])
        axs[0].set_title('$Ee \; vel_x$')
        axs[0].grid()
        axs[1].plot(step_vect, self.ee_vel[0:self.ncalls_get_info_robot, 1],
                    'tab:orange')
        axs[1].set_title('$Ee \; vel_y$')
        axs[1].grid()

        for ax in axs.flat:
            ax.set(xlabel='Steps', ylabel='$[N]$')

        return [axs, fig]
