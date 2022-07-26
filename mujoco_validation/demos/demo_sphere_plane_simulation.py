import os

import matplotlib.pyplot as plt
import mujoco_py
from mujoco_py import MjViewer

import mujoco_validation.src.contact_forces_validation as validate


def start_simulation(model_path):
    """Load the model from the XML and instantiate the Mujoco simulation
    Args:
        path: path of the XML model
    Returns:
        MjSim, MjViewer
    """

    model = mujoco_py.load_model_from_xml(open(model_path).read())
    sim = mujoco_py.MjSim(model)
    viewer = MjViewer(sim)

    return [sim, viewer]


if __name__ == "__main__":

    # Plot and data initializing
    steps = 600

    # Load the model and make a simulator
    model_path = os.path.join(os.path.abspath(os.getcwd()), os.pardir, "config", "sphere_plane.xml")
    [sim, viewer] = start_simulation(model_path)

    # Import the class with the functions needed
    contact_forces_validation = validate.MujocoContactValidation(sim, steps)

    # Simulate and calculate the forces with the explicit method
    for i in range(steps):
        sim.step()
        viewer.render()

        # Calculate contact forces with both built in method and explicit method
        contact_forces_validation.contact_forces(sim)

        # Store results in csv file
        contact_forces_validation.contact_forces_to_csv(sim, "contact_data_simulation")

    # Plot contact forces retrieved by explicit method and explicit method
    contact_forces_validation.plot_contact_forces()
    plt.show()
