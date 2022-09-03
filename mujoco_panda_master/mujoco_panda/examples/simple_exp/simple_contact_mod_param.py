#!/usr/bin/env python

import os
import time
from cmath import tau

import matplotlib.pyplot as plt
import mujoco_py
import numpy as np
from click import prompt
from mujoco_py import MjSim, MjViewer, load_model_from_xml

# Plot and data initializing
xdata = []
ydata = []
k = 0
fz = 0

MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
    <option timestep="0.005" />
    <worldbody>
        <body name="box" pos="0 0 2.1">
            <geom mass = "10" size="1 1 1" type="sphere" condim = "1" solimp="0.99 0.99 0.01" solref="-10000 -200" priority="0"/>
            <joint axis="0 0 1" name="free_ball" type="free" damping="0"/>
        </body>
        <body name="floor" pos="0 0 0">
            <geom mass = "1" condim="1" size="5.0 5.0 1" rgba="0 1 0 1" type="box" solimp="0.99 0.99 0.01" solref="0.01 1" priority="1"/>
        </body>
    </worldbody>
</mujoco>
"""


# Load the model and make a simulator
model = mujoco_py.load_model_from_xml(MODEL_XML)
sim = mujoco_py.MjSim(model)
# viewer = MjViewer(sim)

# Code for changing contact parameters (tau in this case)

tau_vect = np.linspace(
    0.01, 0.1, num=1000
)  # convertito in stiffness da la stiffness falsa
for tau_index in range(tau_vect.shape[0]):

    model.geom_solref[1, 0] = tau_vect[tau_index]
    # print("tau",model.geom_solref[1,0],"damp ratio",model.geom_solref[1,1])

    for i in range(100):
        sim.step()
        # viewer.render()

        contact = sim.data.contact[0]
        dist = contact.dist
        c_array = np.zeros(6, dtype=np.float64)
        mujoco_py.functions.mj_contactForce(sim.model, sim.data, 0, c_array)
        fz = c_array[0]

    # xdata.append(tau_vect[tau_index])
    xdata.append(1 / (tau_vect[tau_index] ** 2))
    # ydata.append(dist/(tau_vect[tau_index]**2)*10/(1-0.99))
    ydata.append(dist)

plt.xlabel("K falsa")
plt.ylabel("Penetration")
plt.plot(xdata, ydata)
plt.show()

input("Press 'Enter' to terminate the execution:")
