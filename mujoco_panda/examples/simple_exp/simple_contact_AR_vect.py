#!/usr/bin/env python

import os
from click import prompt
import mujoco_py
import numpy as np
from mujoco_py import load_model_from_xml, MjSim, MjViewer
import matplotlib.pyplot as plt
import time
import math


# Plot and data initializing
steps = 200
i_dis = np.zeros(steps, dtype=np.float64)
a1 = np.zeros(steps, dtype=np.float64)
v = np.zeros(steps, dtype=np.float64)
x = np.zeros(steps, dtype=np.float64)
a0 = np.zeros(steps, dtype=np.float64)
a1_calc = np.zeros(steps, dtype=np.float64)
pd = np.zeros(steps, dtype=np.float64)
aref = np.zeros(steps, dtype=np.float64)
delta_a = np.zeros(steps, dtype=np.float64)
#AR =  np.zeros(steps, dtype=np.float64)
delta_a_calc = np.zeros(steps, dtype=np.float64)
f_calc = np.zeros(steps, dtype=np.float64)
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
            <geom mass = "1" size="1 1 1" type="sphere" condim = "1" solimp="0.99 0.99 0.01" solref="-10000 -200" priority="0"/>
            <joint axis="0 0 1" name="free_ball" type="free" damping="0"/>
        </body>
        <body name="floor" pos="0 0 0">
            <geom mass = "1" condim="1" size="5.0 5.0 1" rgba="0 1 0 1" type="box" solimp="0.99 0.99 0.01" solref="0.04 1" priority="1"/>
        </body>
    </worldbody>
</mujoco>
"""


# Load the model and make a simulator
model = mujoco_py.load_model_from_xml(MODEL_XML)
sim = mujoco_py.MjSim(model)
#viewer = MjViewer(sim)

# calc b and k

tau = model.geom_solref[1, 0]
damp_ratio = model.geom_solref[1, 1]
d = 0.99
b = 2 / (tau*d)
k = d / ((tau**2) * (damp_ratio ** 2) * (d**2))


x_old = 0
v_old = 0

for i in range(steps):
    i_dis[i] = i
    sim.step()
    # viewer.render()
    ncon = sim.data.ncon
    contact = sim.data.contact[0]
    dist = contact.dist
    c_array = np.zeros(6, dtype=np.float64)
    mujoco_py.functions.mj_contactForce(sim.model, sim.data, 0, c_array)
    fz[i] = c_array[0]

    if ncon != 0:
        x[i] = sim.data.active_contacts_efc_pos
        vel = sim.data.efc_vel[0]
    else:
        x[i] = 0
        vel = 0

    #v[i] = x[i] - x_old
    v[i] = vel

    #[a0[i],pd[i]] = calc_a0(x[i],v[i],a1[i],k,b,d)

    x_old = x[i]
    v_old = v[i]

    # new section
    # linear cost term: J*qacc_unc - aref  (njmax x 1)
    delta_a = sim.data.efc_b

    aref = sim.data.efc_aref  # reference pseudo-acceleration  (njmax x 1)
    AR = sim.data.efc_AR  # J*inv(M)*J' + R (njmax x njmax)
    # qacc = sim.data.qacc_unc # unconstrained acceleration (nv x 1)
    # J = sim.data.efc_J # constraint Jacobian (njmax x nv)
    # Jqacc = J.dot(qacc)
    # pd[i] = -(b*v[i] + k*x[i])
    # delta_a_calc[i] = Jqacc[0] - pd[i] # (nv x 1)
    # print(AR)
    # print(sim.data.efc_AR_rownnz)
    #inv_AR = np.linalg.pinv(AR)
    f_calc = np.linalg.pinv(AR).dot(delta_a)

print(f_calc)


# print(sim.data.qM)
# print(model.nv)
# print(sim.data.efc_diagApprox)
# print(sim.data.efc_R)
# print(model.njmax)


plt.figure()
plt.plot(i_dis, x)


# plt.figure()
# plt.plot(i_dis,v)

# plt.figure()
# plt.plot(i_dis,pd)
# plt.plot(i_dis,aref)

# plt.figure()
# plt.plot(i_dis,delta_a)
# plt.plot(i_dis,delta_a_calc)

plt.figure()
plt.plot(i_dis, -f_calc)
plt.plot(i_dis, fz)

plt.show()
