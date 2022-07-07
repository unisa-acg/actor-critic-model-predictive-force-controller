#!/usr/bin/env python

from cmath import tau
import os
from click import prompt
from cv2 import exp
import mujoco_py
import numpy as np
from mujoco_py import load_model_from_xml, MjSim, MjViewer
import matplotlib.pyplot as plt
import time
import math


# Plot and data initializing
steps = 500
i_dis =  np.zeros(steps, dtype=np.float64)
a1 =  np.zeros(steps, dtype=np.float64)
v =  np.zeros(steps, dtype=np.float64)
x =  np.zeros(steps, dtype=np.float64)
a0 =  np.zeros(steps, dtype=np.float64)
a1_calc =  np.zeros(steps, dtype=np.float64)
pd =  np.zeros(steps, dtype=np.float64)
xdata = []
ydata = []
k = 0
fz = np.zeros(steps, dtype=np.float64)

MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
    <option timestep="0.002" />
    <worldbody>
        <body name="box" pos="0 0 2.1">
            <geom mass = "10" size="1 1 1" type="sphere" condim = "1" solimp="0.99 0.99 0.01" solref="-10000 -200" priority="0"/>
            <joint axis="0 0 1" name="free_ball" type="free" damping="0"/>
        </body>
        <body name="floor" pos="0 0 0">
            <geom mass = "1" condim="1" size="5.0 5.0 1" rgba="0 1 0 1" type="box" solimp="0.99 0.99 0.01" solref="0.04 1" priority="1"/>
        </body>
    </worldbody>
</mujoco>
"""

def calc_a0(x,v,a0,k,b,d,ncon):
    if ncon == 0:
        a0 = 0
    pd = (b*v*d + k*x*d**2)
    #a0 = (a1 + pd) / (1-d)
    a1 = (a0 - pd) * 10
    return [a1,pd]




# Load the model and make a simulator
model = mujoco_py.load_model_from_xml(MODEL_XML)
sim = mujoco_py.MjSim(model)
#viewer = MjViewer(sim)

# calc b and k 

tau = model.geom_solref[1,0]
damp_ratio = model.geom_solref[1,1]
d = 0.99
b = 2 / (tau*d) 
k = 1 / ((tau**2) * (damp_ratio ** 2) * (d**2))
print(k)


x_old = 0
v_old = 0

for i in range(steps):
    i_dis[i] = i
    sim.step()
    #viewer.render()
    ncon = sim.data.ncon
    contact = sim.data.contact[0]
    dist = contact.dist
    c_array = np.zeros(6, dtype=np.float64)
    mujoco_py.functions.mj_contactForce(sim.model, sim.data, 0, c_array)
    fz[i] = c_array[0]
    
    
    if ncon != 0:
        x[i] = sim.data.active_contacts_efc_pos
        qvel = sim.data.qvel[2]
    else:
        x[i] = 0
        qvel = 0

    #v[i] = x[i] - x_old
    v[i] = qvel

    #[a0[i],pd[i]] = calc_a0(x[i],v[i],a1[i],k,b,d)
    [a1_calc[i],pd[i]] = calc_a0(x[i],v[i],9.81,k,b,d,ncon)

    x_old = x[i]
    v_old = v[i]


feq = (b*v*d + k*x*d**2)
    
plt.figure()
plt.plot(i_dis,x)
print(x[499])

plt.figure()
plt.plot(i_dis,v)
print(v[499])

plt.figure()
#plt.plot(i_dis,feq*0)
plt.plot(i_dis,fz)
plt.plot(i_dis,a1_calc)
plt.legend(['fz','a1_calc'])
print(fz[499])
print(feq[499])


plt.figure()
plt.xlabel('')
plt.ylabel('')
plt.plot(i_dis,a1_calc)
plt.plot(i_dis,pd)
plt.legend(['a1','a1_calc','pd'])
plt.show()
