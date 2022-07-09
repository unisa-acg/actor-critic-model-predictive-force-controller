import os
import time
from tkinter import Y
import mujoco_py
import numpy as np
from mujoco_panda import PandaArm
from mujoco_panda.utils.viewer_utils import render_frame
from mujoco_panda.utils.debug_utils import ParallelPythonCmd
from mujoco_panda.controllers.torque_based_controllers import OSHybridForceMotionController
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.sans-serif'] = "Comic Sans MS"
import logging

logging.getLogger('matplotlib.font_manager').disabled = True

plt.ion()
class DynamicUpdate():
    #Suppose we know the x range
    min_x = 0
    max_x = 10

    def on_launch(self):
        #Set up plot
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([],[], 'o')
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        self.ax.set_autoscalex_on(True)
        #self.ax.set_xlim(self.min_x, self.max_x)
        #Other stuff
        self.ax.grid()
        ...

    def on_running(self, xdata, ydata):
        #Update data (with the new _and_ the old points)
        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    # def __call__(self):
    #     import numpy as np
    #     import time
    #     self.on_launch()
    #     xdata = []
    #     ydata = []
    #     for x in np.arange(0,10,0.5):
    #         xdata.append(x)
    #         ydata.append(np.exp(-x**2)+10*np.exp(-(x-7)**2))
    #         self.on_running(xdata, ydata)
    #         time.sleep(1)
    #     return xdata, ydata

plt.xlabel('Steps')
plt.ylabel('Penetration distance')
xdata = []
ydata = []
k = 0
dist = 0
i_cont = 0
plt.autoscale(enable=True, axis='both', tight=None)
steps = 0
i = 0
count = 0

def exec_func(cmd):
    if cmd == '':
        return None
    a = eval(cmd)
    print(cmd)
    print(a)
    if a is not None:
        return str(a)

MODEL_PATH = os.environ['MJ_PANDA_PATH'] + \
    '/mujoco_panda/models/'


# controller parameters
KP_P = np.array([7000., 7000., 7000.])
KP_O = np.array([3000., 3000., 3000.])
ctrl_config = {
    'kp_p': KP_P,
    'kd_p': 2.*np.sqrt(KP_P),
    'kp_o': KP_O,  # 10gains for orientation
    'kd_o': [1., 1., 1.],  # gains for orientation
    'kp_f': [1., 1., 1.],  # gains for force
    'kd_f': [0., 0., 0.],  # 25gains for force
    'kp_t': [1., 1., 1.],  # gains for torque
    'kd_t': [0., 0., 0.],  # gains for torque
    'alpha': 3.14*0,
    'use_null_space_control': True,
    'ft_dir': [0, 0, 0, 0, 0, 0],
    # newton meter
    'null_kp': [5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0],
    'null_kd': 0,
    'null_ctrl_wt': 2.5,
    'use_orientation_ctrl': True,
    'linear_error_thr': 0.025,
    'angular_error_thr': 0.01,
}

if __name__ == "__main__":
    p = PandaArm(model_path=MODEL_PATH+'panda_block_table.xml',
                 render=True, compensate_gravity=True, smooth_ft_sensor=True)

    if mujoco_py.functions.mj_isPyramidal(p.model):
        print("Type of friction cone is pyramidal")
    else:
        print("Type of friction cone is eliptical")

    # cmd = ParallelPythonCmd(exec_func)
    p.set_neutral_pose()
    p.step()
    time.sleep(1.0)
    
    # create controller instance with default controller gains
    ctrl = OSHybridForceMotionController(p, config=ctrl_config)    

    # SINGOLA TRAIETTORIA

        # --- define trajectory in position -----
    curr_ee, curr_ori = p.ee_pose()
    goal_pos_1 = curr_ee.copy()
    goal_pos_1[2] = 0.2
    goal_pos_1[0] += 0
    goal_pos_1[1] -= 0
    target_traj_1 = np.linspace(curr_ee, goal_pos_1, 100)
    z_target = curr_ee[2]
    #target_ori = np.asarray([0, -0.924, -0.383, 0], dtype=np.float64)
    target_traj = target_traj_1
    target_ori = curr_ori
    # FINE SINGOLA TRAIETTORIA

    ctrl.set_active(True) # activate controller (simulation step and controller thread now running)

    now_r = time.time()
    sim = p.sim

    while steps < 2000:
        
        while i < target_traj.shape[0]:
            # get current robot end-effector pose
            robot_pos, robot_ori = p.ee_pose() 
            elapsed_r = time.time() - now_r

            # render controller target and current ee pose using frames
            render_frame(p.viewer, robot_pos, robot_ori)
            render_frame(p.viewer, target_traj[i, :], target_ori, alpha=0.2)
            #ctrl.set_goal(target_traj[i, :], target_ori)

            if i == 130: # activate force control when the robot is near the table
                ctrl.change_ft_dir([0,0,1,0,0,0]) # start force control along Z axis
                ctrl.set_goal(target_traj[i, :],
                            target_ori, goal_force=[0, 0, -25]) # target force in cartesian frame
            else:
                ctrl.set_goal(target_traj[i, :], target_ori)


            i += 1 # change target less frequently compared to 
            contact_forces_ee = ctrl.contact_forces_ee()


            print('number of contacts', sim.data.ncon)
            contact = sim.data.contact[0]
            print('contact number',0)
            print('dist', contact.dist)
            print('')
            c_array = np.zeros(6, dtype=np.float64)
            mujoco_py.functions.mj_contactForce(sim.model, sim.data, 0, c_array)
            print('c_array', c_array)

            xdata.append(i_cont)
            ydata.append(contact.dist)
            #dist_plot.on_running(xdata, ydata)
            contact.dist = 0
            i_cont += 1
            p.render() # render the visualisation
            
        if i == target_traj.shape[0]:
            break
        
        # ctrl.set_active(False)
        # #ctrl.stop_controller_cleanly()
        # p.step()
        # contact = sim.data.contact[0]
        # mujoco_py.functions.mj_contactForce(sim.model, sim.data, 0, c_array)
        # xdata.append(i_cont)
        # ydata.append(contact.dist)
        # i_cont += 1
        # steps += 1
        print(steps)

    plt.plot(xdata,ydata)
    plt.show()
    # print(xdata)
    # print(ydata)
    plt.savefig('compliance robot.png')
    
    #input("Trajectory complete. Hit Enter to deactivate controller")
    ctrl.set_active(False)
    #ctrl.stop_controller_cleanly()
    print('number of contacts', sim.data.ncon)
    print("equilibrium penetration = ", contact.dist)
    print("equilibrium force= ",c_array[0])
    print ("smoothed FT reading: ", p.get_ft_reading(pr=True))


