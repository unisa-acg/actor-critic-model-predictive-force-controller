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
steps = 500
xdata = []
ydata = []
dist_vect = []
dist_punto_vect = []
vel_z_vect = []
vel_deformation = []
fz_vect = []
F_eq_vect = []
ratio_vect = []
ratio_dis = []
ratio_forces_vect = []
pos_z_vect = []
residual_vect = np.zeros(steps, dtype=np.float64)
a0_vect = np.zeros(steps, dtype=np.float64)
k = 0
fz = 0
dist_old = 0
residual_old = 0
dresidual_old = 0
kk = 0



MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
    <option timestep="0.005" />
    <worldbody>
        <body name="box" pos="0 0 3">
            <geom mass = "1" size="1 1 1" type="sphere" condim = "1" solimp="0.99 0.99 0.01" solref="0.0001 1" priority="0"/>
            <joint axis="0 0 1" name="free_ball" type="free" damping="0"/>
        </body>
        <body name="floor" pos="0 0 0">
            <geom mass = "0" condim="1" size="5.0 5.0 1" rgba="0 1 0 1" type="box" solimp="0.99 0.99 0.01" solref="0.1 1" priority="1"/>
        </body>
    </worldbody>
</mujoco>
"""

#solref -10000 -200
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

    K_eq = np.abs(model.geom_solref[1,0]) * 1 * 1/(1-0.99)
    R_eq = np.abs(model.geom_solref[1,1]) * 1 * 1/(1-0.99)
    print(K_eq)
    
    # from tau to b and k. b damping, k stiffness
    tau = model.geom_solref[1,0]
    damp_ratio = model.geom_solref[1,1]
    d = 0.99
    b = 2 / (tau*d) 
    k = 1 / ((tau**2) * (damp_ratio ** 2) * (d**2))
    


    for i in range(steps):
        sim.step()
        viewer.render()

        contact = sim.data.contact[0]
        dist = np.abs(contact.dist)
        c_array = np.zeros(6, dtype=np.float64)
        mujoco_py.functions.mj_contactForce(sim.model, sim.data, 0, c_array)
        fz = c_array[0]
        qvel = sim.data.qvel
        vel_z = qvel[2]
        
        # cal velocity of deformation
        dist_new = dist
        dist_punto = dist_new - dist_old
        dist_punto_vect.append(dist_punto)

        #calc equivalent force
        F_eq = K_eq * np.abs(dist) + R_eq * (dist_punto)
        ratio_forces_vect.append(F_eq / fz)
        #calc R_eq backward 
        if dist_punto != 0:
            R_eq_calc = (fz - K_eq * np.abs(dist_new))/(dist_punto)
            ratio_vect.append(R_eq_calc / np.abs(model.geom_solref[1,1]))
        else:
            ratio_vect.append(0)
        
        #retrieve position of ball
        qpos = sim.data.qpos
        pos_z = qpos[2]
        pos_z_vect.append(pos_z)

        if sim.data.ncon != 0:
            residual_new = sim.data.active_contacts_efc_pos
        else:
            residual_new = 0

        print(residual_new)
        residual_vect[i] = residual_new 

        dresidual_new = residual_new - residual_old
        a1 = dresidual_new - dresidual_old 

        a0 = (a1 + d*(b*dresidual_new + k*residual_new)) / (1-d)

        a0_vect[i] = a0

        dist_old = dist_new
        residual_old = residual_new
        dresidual_old = dresidual_new


        xdata.append(i)
        
        fz_vect.append(fz)
        vel_z_vect.append(vel_z)
        dist_vect.append(dist)
        F_eq_vect.append(F_eq)

    #print(fz)
    #print(dist)

    
    ##xdata.append(tau_vect[tau_index])
    #xdata.append(1/(tau_vect[tau_index]**2))
    ##ydata.append(dist/(tau_vect[tau_index]**2)*10/(1-0.99))
    #ydata.append(dist)

print(residual_vect)
print(a0_vect)

for k in range(len(dist_vect)-1):
    vel_deformation.append(dist_vect[k+1]-dist_vect[k]) 
vel_deformation.append(vel_deformation[k])

#print(ratio_vect)

plt.figure()
plt.plot(xdata,ratio_vect)

plt.figure()
plt.plot(xdata,residual_vect)
plt.ylabel('Residual')

plt.figure()
plt.plot(xdata,a0_vect)
plt.ylabel('a0')



plt.figure()
plt.plot(xdata,ratio_forces_vect)

plt.figure()
plt.plot(xdata,pos_z_vect)
alpha = 0.005
x = np.linspace(0, i, i+1)
y = np.exp(-alpha*x) + 2
plt.plot(xdata,y)

plt.figure()
#plot penetration
plt.subplot(2, 2, 1)
plt.xlabel('Step')
plt.ylabel('Penetration')
plt.plot(xdata,dist_vect)
plt.grid()

#plot velocity
plt.subplot(2, 2, 2)
plt.xlabel('Step')
plt.ylabel('Velocity')
plt.plot(xdata,dist_punto_vect)
plt.grid()

#plot contact force
plt.subplot(2, 2, 3)
plt.xlabel('Step')
plt.ylabel('Contact Force')
plt.plot(xdata,fz_vect)
plt.grid()

#plot equivalent force
plt.subplot(2, 2, 4)
plt.xlabel('Step')
plt.ylabel('Equivalent force')
plt.plot(xdata,F_eq_vect)
plt.grid()

plt.show(block = False)
plt.pause(3)
plt.close("all")