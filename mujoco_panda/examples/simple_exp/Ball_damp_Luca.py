#!/usr/bin/env python

from cmath import tau
import os
from click import prompt
import mujoco_py
import numpy as np
from mujoco_py import load_model_from_xml, MjSim, MjViewer
import matplotlib.pyplot as plt
import time


# Plot and data initializing
xdata = []
d = []
v = []
force = []
teoric_f = []
contact_vel = []
k = 0
fz = 0
dist_old = 0
vel_dist_old = 0
ball_pos = []
ball_vel = []

MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
    <option timestep="0.005" />
    <worldbody>
        <body name="box" pos="0 0 3">
            <geom mass = "5" size="1 1 1" type="sphere" condim = "1" solimp="0.99 0.99 0.01" solref="-10000 -200" priority="0"/>
            <joint axis="0 0 1" name="free_ball" type="free" damping="0"/>
        </body>
        <body name="floor" pos="0 0 0">
            <geom mass = "1" condim="1" size="5.0 5.0 1" rgba="0 1 0 1" type="box" solimp="0.99 0.99 0.01" solref="-100 -20" priority="1"/>
        </body>
    </worldbody>
</mujoco>
"""


# Load the model and make a simulator
model = mujoco_py.load_model_from_xml(MODEL_XML)
sim = mujoco_py.MjSim(model)
viewer = MjViewer(sim)


# Code for changing contact parameters (tau in this case)

#tau_vect = np.linspace(0.01, 0.1, num=1000) #convertito in stiffness da la stiffness falsa
tau_vect = np.array([0])
for tau_index in range(tau_vect.shape[0]):

    #model.geom_solref[1,0] = tau_vect[tau_index]
    
    print("tau",model.geom_solref[1,0],"damp ratio",model.geom_solref[1,1])

    for i in range(500):
        sim.step()
        viewer.render()
        ncon = sim.data.ncon

        contact = sim.data.contact[0]
        dist = contact.dist
        # Derivata della dist
        vel_dist = dist - dist_old
        dist_old = dist
        contact_vel.append(vel_dist)
        # Derivata della vel
        acc_dist = vel_dist - vel_dist_old
        vel_dist_old = vel_dist

        c_array = np.zeros(6, dtype=np.float64)
        mujoco_py.functions.mj_contactForce(sim.model, sim.data, 0, c_array)
        fz = c_array[0]

        if ncon != 0:
            pos = sim.data.active_contacts_efc_pos[0]
        else:
            pos = 0

        vel = sim.data.efc_vel[0]
        
        #t_force = model.geom_solref[1,0]*dist*100 + 200*model.geom_solref[1,1]*vel_dist*100

        if dist == 0:
            t_force = 0
        else:
            t_force = 0.99*(model.geom_solref[1,0]*dist*100 + 2*model.geom_solref[1,1]*vel_dist*100) + 9.81
        
        real_k = model.geom_solref[1,0]*dist*100

        d.append(dist)
        xdata.append(i)
        v.append(vel)
        force.append(fz)
        teoric_f.append(t_force)
        ball_pos.append(pos)
        ball_vel.append(vel)
    
    damp = 9.81/vel_dist

    print('damping', damp)
    print('real k', real_k)    
    #xdata.append(tau_vect[tau_index])
    #xdata.append(1/(tau_vect[tau_index]**2))
    #ydata.append(dist/(tau_vect[tau_index]**2)*10/(1-0.99))
    #ydata.append(dist)

a = np.asarray(ball_pos)
a.tofile('ball_pos.csv',sep=',')

a = np.asarray(ball_vel)
a.tofile('ball_vel.csv',sep=',')

a = np.asarray(force)
a.tofile('force.csv',sep=',')
# plt.figure(1)
# plt.xlabel('i')
# plt.ylabel('dist')
# plt.plot(xdata,d)
# plt.grid(True)

# plt.figure(2)
# plt.xlabel('i')
# plt.ylabel('vel')
# plt.plot(xdata,v)
# plt.grid(True)

# plt.figure(3)
# plt.xlabel('i')
# plt.ylabel('vel')
# plt.plot(xdata,contact_vel)
# plt.grid(True)

# plt.figure(4)
# plt.xlabel('i')
# plt.ylabel('force')
# plt.plot(xdata,force)
# plt.grid(True)

# plt.figure(5)
# plt.plot(xdata,teoric_f)
# plt.grid(True)
# plt.legend(['Teoric force'])
# plt.show()

input("Press 'Enter' to terminate the execution:")