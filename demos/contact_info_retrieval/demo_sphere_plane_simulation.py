#!/usr/bin/env python

import os
from click import prompt
import mujoco_py
import numpy as np
from mujoco_py import load_model_from_xml, MjSim, MjViewer
import matplotlib.pyplot as plt
import time
import math
import mujoco_validation.contact_forces_validation as validate


def load_model_start_sim():
    # XML model definition
    MODEL_XML = """
    <?xml version="1.0" ?>
    <mujoco>
        <option timestep="0.002" />
        <worldbody>
            <body name="sphere" pos="0 0 3">
                <geom mass = "5" size="1 1 1" type="sphere" condim = "1" solimp="0.99 0.99 0.01" solref="0.01 1" priority="0" name="sphere"/>
                <joint axis="0 0 1" name="free_ball" type="free"/>
            </body>
            <body name="floor" pos="0 0 0">
                <geom mass = "1" condim="1" size="5.0 5.0 1" rgba="0 1 0 1" type="box" solimp="0.99 0.99 0.01" solref="-100 -20" priority="1" name="floor"/>
            </body>
        </worldbody>
    </mujoco>
    """
    model = mujoco_py.load_model_from_xml(MODEL_XML)
    sim = mujoco_py.MjSim(model)
    viewer = MjViewer(sim)

    return [sim, viewer]


if __name__ == "__main__":

    # Plot and data initializing
    steps = 600
    i_dis = np.zeros(steps, dtype=np.float64)

    # Load the model and make a simulator
    [sim, viewer] = load_model_start_sim()

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
