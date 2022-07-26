#!/usr/bin/env python

import math
import os
import time

import matplotlib.pyplot as plt
import mujoco_py
import numpy as np
import validate as validate
from click import prompt
from mujoco_py import MjSim, MjViewer, load_model_from_xml

if __name__ == "__main__":

    # Plot and data initializing
    steps = 600
    i_dis = np.zeros(steps, dtype=np.float64)
    ncon = np.zeros(steps, dtype=np.float64)
    xdata = []
    ydata = []
    k = 0
    fz = np.zeros(steps, dtype=np.float64)

    # XML Model
    MODEL_XML = """
    <?xml version="1.0" ?>
    <mujoco>
        <option timestep="0.002" />
        <worldbody>
            <body name="box" pos="-2.7 -2.7 3">
                <geom mass = "1" size="1 1 1" type="box" condim = "1" solimp="0.99 0.99 0.01" solref="0.01 1" priority="0" name="box"/>
                <joint axis="0 0 1" name="free_ball" type="free"/>
            </body>
            <body name="box1" pos="2.7 2.7 3.5">
                <geom mass = "10" size="1 1 1" type="box" condim = "1" solimp="0.99 0.99 0.01" solref="0.01 1" priority="0" name="box1"/>
                <joint axis="0 0 1" name="free_ball1" type="free"/>
            </body>
            <body name="floor" pos="0 0 0.5">
                <geom mass = "1" condim="1" size="5.0 5.0 1" rgba="0 1 0 1" type="box" solimp="0.99 0.99 0.01" solref="0.05 1" priority="1" name="floor"/>
            </body>
        </worldbody>
    </mujoco>
    """

    # Load the model and make a simulator
    model = mujoco_py.load_model_from_xml(MODEL_XML)
    sim = mujoco_py.MjSim(model)
    viewer = MjViewer(sim)

    # Import the class with the functions needed
    Validate = validate.MujocoContactValidation(sim, steps)

    aref = np.zeros(steps, dtype=np.float64)
    dist_vect = np.zeros(steps, dtype=np.float64)
    a0 = np.zeros(steps, dtype=np.float64)
    # Simulate and calculate the forces with the explicit method
    for i in range(steps):
        i_dis[i] = i
        sim.step()
        viewer.render()

        # Calculate contact forces with both built in method and explicit method
        aref_vect = Validate.contact_forces(sim)

        aref[i] = sim.data.efc_aref[0]
        a0[i] = sim.data.efc_b[0] + aref[i]
        if sim.data.ncon != 0:
            dist = sim.data.contact[0].dist
            dist_vect[i] = dist
        # Store results in csv file

        Validate.contact_forces_to_csv(sim)
        Validate.contact_forces_to_csv_panda(sim)

    # Plot contact forces retrieved by explicit method and explicit method
    Validate.plot_contact_forces()

    plt.figure()
    plt.plot(i_dis, aref)
    plt.grid()

    plt.figure()
    plt.plot(i_dis, a0)
    plt.grid()

    plt.show()
