import csv
import os.path
from datetime import datetime
from itertools import combinations

import matplotlib.pyplot as plt
import mujoco_py
import numpy as np


class MujocoContactValidation:
    """
    Class that contains methods to compute, plot and store in a .csv the contact forces
    and information happening during a MuJoCo simulation
    """

    def __init__(self, sim, steps):
        """
        Instantiate the variables needed for the other methods

        Args:
            sim (mjSim): MuJoCo simulator istance
            steps: total steps of the simulation

        Returns:
            NDArray: (A+R)^-1 component
        """
        self.date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.ncalls = 0
        self.ncallspanda = 0
        self.body_combinations = [_ for _ in combinations(sim.model.body_names, 2)]
        self.combinations_forces = np.zeros(
            (len(self.body_combinations), steps), dtype=np.float64, order="C"
        )
        self.combinations_forces_built_in = np.zeros(
            (len(self.body_combinations), steps), dtype=np.float64, order="C"
        )
        self.contact_forces_calls = 0
        self.contact_forces_to_csv_call = 0
        self.forces_vect_mujocoenv = np.zeros((6, steps), dtype=np.float64, order="C")

    # ---------------------------------------------------------------------------- #
    #                          Contact forces calculation                          #
    # ---------------------------------------------------------------------------- #

    # ------------------------------- Subfunctions ------------------------------- #

    def get_inv_AR(self, sim, efc_addresses):
        """Compute the (A+R)^-1 component

        Args:
            sim (mjSim): MuJoCo simulator istance at current timestep
            efc_addresses (NDArray): vector that stores the indices of the contact
                points

        Returns:
            NDArray: (A+R)^-1 component
        """

        M = np.ndarray(shape=(len(sim.data.qvel) ** 2,), dtype=np.float64, order="C")
        mujoco_py.functions.mj_fullM(sim.model, M, sim.data.qM)
        M = np.reshape(M, (len(sim.data.qvel), len(sim.data.qvel)))
        J = sim.data.efc_J
        J_invM = J.dot(np.linalg.inv(M))
        A = J_invM.dot(np.transpose(J))
        max_efc = max(efc_addresses)
        R = np.diag(sim.data.efc_R)
        AR = A + R

        # Invert only the "active" part of the matrix,
        # the rest is full of zeros and not of interest
        AR_mod_efc = AR[: max_efc + 1, : max_efc + 1]
        inv_AR_efc = np.linalg.pinv(AR_mod_efc)
        inv_AR = np.zeros((sim.model.njmax, sim.model.njmax), dtype=np.float64)
        inv_AR[: max_efc + 1, : max_efc + 1] = inv_AR_efc

        return inv_AR

    def get_delta_a(self, sim):
        """Compute the (a0 - aref) components

        Args:
            sim (mjSim): MuJoCo simulator instance at current timestep

        Returns:
            NDArray: vector containing the difference between a0 and aref of each
                contact
            NDArray: vector containing the addresses of each contact
            NDArray: vector containing all the possible combinations of the bodies
            NDArray: vector containing the tuple list of the bodies names in contact
            NDArray: vector of zeroes with the correct dimensions that will contain the
                combinations of forces
            NDArray: vector of zeroes with the correct dimensions that will contain the
                combinations of forces obtained with the built_in method
            NDArray: vector containing the normal force of each contact
        """

        # ----------------------------- Data initializing ---------------------------- #
        ncon = sim.data.ncon  # number of contacts
        model = sim.model  # shortcut to simulation model data
        data = sim.data  # shortcut to simulation data
        njmax = model.njmax  # max number of steps

        dist_njmax = np.zeros(njmax, dtype=np.float64)
        c_array = np.zeros(6, dtype=np.float64)
        efc_addresses = np.zeros(ncon, dtype=np.int32)
        normal_forces = np.zeros(ncon, dtype=np.float64)

        combinations_forces = np.zeros(len(self.body_combinations), dtype=np.float64)
        combinations_forces_built_in = np.zeros(
            len(self.body_combinations), dtype=np.float64
        )

        if self.combinations_forces.shape[0] != len(combinations_forces):
            self.combinations_forces = np.zeros(
                len(combinations_forces), dtype=np.float64
            )
            self.combinations_forces = np.transpose(self.combinations_forces)
            self.combinations_forces_built_in = np.zeros(
                len(combinations_forces_built_in), dtype=np.float64
            )
            self.combinations_forces_built_in = np.transpose(
                self.combinations_forces_built_in
            )
        contact_list_tuple = [0] * njmax
        # ---------------------------------------------------------------------------- #

        for i in range(ncon):

            # Retrieve penetration, velocity and parameters for each contact (k, b, d)
            contact = data.contact[i]
            efc_address = contact.efc_address
            efc_addresses[i] = efc_address
            dist_njmax[efc_address] = data.active_contacts_efc_pos[efc_address]
            geom1_name = model.geom_id2name(contact.geom1)
            geom2_name = model.geom_id2name(contact.geom2)
            geom_tuple = (geom1_name, geom2_name)
            contact_list_tuple[efc_address] = geom_tuple

            # Retrieve also the contact forces with built-in functions for comparison
            mujoco_py.functions.mj_contactForce(model, data, i, c_array)
            normal_forces[efc_address] = c_array[0]

        vel_njmax = data.efc_vel
        k_vect = data.efc_KBIP[:, 0]
        b_vect = data.efc_KBIP[:, 1]

        # Compute reference acceleration a_ref
        a_ref = [
            -(k_vect[k] * dist_njmax[k] + b_vect[k] * vel_njmax[k])
            for k in range(len(k_vect))
        ]

        # Compute unconstrained acceleration a_0
        qacc = data.qacc_unc  # unconstrained acceleration in joint space (nv x 1)
        J = data.efc_J  # constraint Jacobian (njmax x nv)
        a_0 = J.dot(qacc)  # (njmax x nv) x (nv x 1) = (njmax x 1)

        # Compute delta_a
        delta_a = a_0 - a_ref  # (njmax x 1)

        return [
            delta_a,
            efc_addresses,
            self.body_combinations,
            contact_list_tuple,
            combinations_forces,
            combinations_forces_built_in,
            normal_forces,
        ]

    # ------------------------------- Main functions ----------------------------- #

    def contact_forces(self, sim):
        """Given a MuJoCo mjSim simulation, returns the vector of contact forces,
        following the formula ``f = (A + R)^-1 * (aref - a0)``

        Args:
            sim (mjSim): MuJoCo simulation instance at current timestep

        Returns:
            NDArray: vector of contact forces (with explicit method)
            NDArray: vector of contact forces (with built-in method)
            NDArray: vector of combinations of bodies in the simulation
        """

        # Inizialization of vectors and data
        data = sim.data
        model = sim.model
        njmax = model.njmax
        ncon = data.ncon
        inv_AR = np.zeros(njmax, dtype=np.float64)

        if ncon != 0:

            # Compute (a_ref - a_0)
            [
                delta_a,
                efc_addresses,
                body_combinations,
                contact_list_tuple,
                combinations_forces,
                combinations_forces_built_in,
                normal_forces,
            ] = self.get_delta_a(sim)

            # Compute (A + R)^-1
            inv_AR = self.get_inv_AR(sim, efc_addresses)

            # Compute the vector of total contact forces
            contact_forces_vect = -inv_AR.dot(delta_a)

            # Forces between pair of bodies are stored as instance members
            for ind_efc in efc_addresses:
                contact_tuple = contact_list_tuple[ind_efc]
                ind_tuple = body_combinations.index(contact_tuple)
                combinations_forces[ind_tuple] = (
                    combinations_forces[ind_tuple] + contact_forces_vect[ind_efc]
                )
                # Do the same for forces retrieved via built-in functions
                combinations_forces_built_in[ind_tuple] = (
                    combinations_forces_built_in[ind_tuple] + normal_forces[ind_efc]
                )

            self.combinations_forces[:, self.contact_forces_calls] = combinations_forces

            # Store the forces with mujoco built-in method for comparison
            self.combinations_forces_built_in[
                :, self.contact_forces_calls
            ] = combinations_forces_built_in

        self.contact_forces_calls += 1

        return [
            self.combinations_forces_built_in,
            self.combinations_forces,
            self.body_combinations,
        ]

    def plot_contact_forces(self):
        """Plot the contact forces happened between all the pair of bodies that came in
        contact during the simulation.
        Must be called AFTER ``contact_forces()`` method"""

        step_vect = np.arange(0, self.contact_forces_calls, 1, dtype=int)

        for i in range(len(self.body_combinations)):
            is_all_zero = np.all((self.combinations_forces[i, :] == 0))
            if not is_all_zero:
                plt.figure()
                str = " and ".join(self.body_combinations[i])
                plt.title(r"Contact force between " + str + " (Explicit Method)")
                plt.xlabel(r"Steps")
                plt.ylabel(r"Total contact force $[N]$")
                plt.grid()
                plt.plot(
                    step_vect,
                    self.combinations_forces[i, : self.contact_forces_calls],
                    linewidth=2,
                )

                plt.figure()
                str = " and ".join(self.body_combinations[i])
                plt.title(r"Contact force between " + str + " (Built-in Method)")
                plt.xlabel(r"Steps")
                plt.ylabel(r"Total contact force $[N]$")
                plt.grid()
                plt.plot(
                    step_vect,
                    self.combinations_forces_built_in[i, : self.contact_forces_calls],
                    linewidth=2,
                    color="red"
                )

        # TODO: solve issue where if plt.show() is called here, then the rest of the
        # code won't execute until you close the figure
        # plt.show()

    # ---------------------------------------------------------------------------- #
    #                       Save contacts informations to csv                      #
    # ---------------------------------------------------------------------------- #

    def contact_forces_to_csv(self, sim, csv_name):
        """Given a MuJoCo Sim simulation prints all the contacts parameters and forces
        registered into a .csv file in the current location

        Args:
            sim (mjSim): MuJoCo simulator instance at current timestep
        """

        # Initialization of data
        data = sim.data
        model = sim.model
        ncon = data.ncon

        # Store contact info in buffer, to be appended to main file containing all the
        # steps.
        mujoco_py.functions.mj_printData(model, data, "buffer.txt")

        # Retrieve only the lines of interest -> contact struct lines
        with open("buffer.txt", "r") as file:
            lines = file.readlines()
            index_begin_contact_mjdata = lines.index("CONTACT\n")
            index_end_contact_mjdata = lines.index("EFC_TYPE\n")
            contact_lines = lines[index_begin_contact_mjdata:index_end_contact_mjdata]
            contact_lines[0] = (
                contact_lines[0] + "Step " + str(self.contact_forces_to_csv_call)
            )
        os.remove("buffer.txt")

        contact_forces_string = ["contact_forces"]

        # Write on main file the info of contact structure, including the contact forces
        # retrieved by function mj_contact_forces

        self.contact_forces_to_csv_call += 1

        if ncon != 0:
            # with open("contact_data_simulation.csv", "a") as f:
            with open("../output/{}_{}.csv".format(csv_name, self.date_time), "a") as f:
                writer = csv.writer(f)

                # populate list with lines that we want to exclude from contact
                # structure
                lines_to_exclude = ["exclude", "includemargin", "pos", "frame"]

                for i in range(len(contact_lines)):
                    line_split = contact_lines[i].split()

                    # exclude the line if present in list of exclusion
                    flag = any(line in line_split for line in lines_to_exclude)
                    if flag is True:
                        continue

                    # add contact forces line to the structure to be written to file
                    if "dim" in line_split:
                        linesplit_check = contact_lines[i - 1].split()
                        for k in range(ncon):
                            if str(k) in linesplit_check[0]:
                                index_contact = k
                                contact_forces_array = np.zeros(6, dtype=np.float64)

                                mujoco_py.functions.mj_contactForce(
                                    model, data, index_contact, contact_forces_array
                                )

                                for kk in range(len(contact_forces_array)):
                                    contact_forces_string.append(
                                        contact_forces_array[kk]
                                    )
                                writer.writerow(contact_forces_string)
                                contact_forces_string = ["contact_forces"]

                        writer.writerow(line_split)
                    else:
                        writer.writerow(line_split)

            f.close()
