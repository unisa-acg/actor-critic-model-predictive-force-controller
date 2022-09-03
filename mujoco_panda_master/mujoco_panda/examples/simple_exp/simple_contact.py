#!/usr/bin/env python

import os
import time

import matplotlib.pyplot as plt
import mujoco_py
import numpy as np
from click import prompt
from mujoco_py import MjSim, MjViewer, load_model_from_xml

plt.ion()


class DynamicUpdate:
    # Suppose we know the x range
    min_x = 0
    max_x = 10

    def on_launch(self):
        # Set up plot
        self.figure, self.ax = plt.subplots()
        (self.lines,) = self.ax.plot([], [], "o")
        # Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        self.ax.set_autoscalex_on(True)
        # self.ax.set_xlim(self.min_x, self.max_x)
        # Other stuff
        self.ax.grid()
        ...

    def on_running(self, xdata, ydata):
        # Update data (with the new _and_ the old points)
        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)
        # Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        # We need to draw *and* flush
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


# Plot and data initializing
c_force_plot = DynamicUpdate()
c_force_plot.on_launch()
plt.xlabel("Steps")
plt.ylabel("Contact Force")
xdata = []
ydata = []
k = 0
fz = 0

## Plot the force retrieved by elastic model
c_force_el_plot = DynamicUpdate()
c_force_el_plot.on_launch()
plt.xlabel("Steps")
plt.ylabel("Elastic Force")
xdata_el = []
ydata_el = []

MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
    <option timestep="0.005" />
    <worldbody>
        <body name="box" pos="0 0 2.3">
            <geom mass = "10" size="1 1 1" type="sphere" condim = "3" solimp="0.99 0.99 0.01" solref="0.0001 1" priority="0"/>
            <joint axis="0 0 1" name="free_ball" type="free"/>
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
viewer = MjViewer(sim)

# Start simulation
for i in range(100):
    sim.step()
    viewer.render()
    print("number of contacts", sim.data.ncon)

    for ii in range(sim.data.ncon):
        contact = sim.data.contact[ii]
        print("dist", contact.dist)
        c_array = np.zeros(6, dtype=np.float64)
        mujoco_py.functions.mj_contactForce(sim.model, sim.data, ii, c_array)
        print("c_array", c_array)
        fz = c_array[0]

        # Plot the data if contact registered
        if fz != 0:
            xdata.append(k)
            ydata.append(fz)
            ydata[0] = 0
            c_force_plot.on_running(xdata, ydata)
            k = k + 1

        for kk in range(500):
            if sim.data.efc_KBIP[kk, 1] != 0:
                # print(sim.data.efc_KBIP[kk])
                # print(contact.solref)
                # print(contact.solimp)
                # print(contact.pos)

                # Add marker of contact point to the renderer
                viewer.add_marker(
                    pos=np.array([contact.pos[0], contact.pos[1], contact.pos[2]])
                )
                viewer.render()

                # Prove correlation of elastic model
                dist = contact.dist
                stiffness = 1 / (contact.solref[0] ** 2)
                imped = contact.solimp[0]
                mass_sphere = sim.model.body_mass[1]
                f_el = dist * stiffness * mass_sphere / (1 - imped)
                vect_res = [dist, stiffness, mass_sphere, f_el]
                print("penetr, k, m_sphere, force = ", vect_res)
                print(stiffness * dist * mass_sphere)

                xdata_el.append(k)
                ydata_el.append(f_el)
                c_force_el_plot.on_running(xdata_el, ydata_el)

print("")
print("At steady state we have:")
print("Contact force =", fz, "N")
print("Elastic force =", f_el, "N")
print("Penetration =", dist * 10**3, "mm")
print("")

input("Press 'Enter' to terminate the execution:")
