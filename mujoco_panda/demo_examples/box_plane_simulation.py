#!/usr/bin/env python

import os
from click import prompt
import mujoco_py
import numpy as np
from mujoco_py import load_model_from_xml, MjSim, MjViewer
import matplotlib.pyplot as plt
import time
import math
import validate as validate


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

    # Simulate and calculate the forces with the explicit method
    for i in range(steps):
        i_dis[i] = i
        sim.step()
        viewer.render()

        # Calculate contact forces with both built in method and explicit method
        Validate.contact_forces(sim)

        # Store results in csv file
        Validate.contact_forces_to_csv(sim)

    # Plot contact forces retrieved by explicit method and explicit method
    Validate.plot_contact_forces()
    plt.show()
