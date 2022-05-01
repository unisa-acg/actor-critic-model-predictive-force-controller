#
import csv
import math
import os.path
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.pyplot import ion, show
import mujoco_py
import numpy as np
import pandas as pd
from click import prompt


class MujocoContactValidation:
    def __init__(self, sim, steps):
        self.ncalls = 0
        body_names = sim.model.body_names
        self.combinations_vect = [i for i in combinations(body_names, 2)]
        self.combinations_forces_vect = np.zeros(
            (len(self.combinations_vect), steps), dtype=np.float64, order="C"
        )
        self.combinations_forces_built_in = np.zeros(
            (len(self.combinations_vect), steps), dtype=np.float64, order="C"
        )
        self.contact_forces_calls = 0
        self.steps = steps

    def contact_forces(self, sim):
        """Given a MuJoCo mjSim simulation,returns the vector of contact forces, following the formula f = (A + R)^-1 * (aref - a0). Moreover, it returns the contacts parameters vectors (b,k,d)"""

        # Inizialization of vectors and data
        data = sim.data
        model = sim.model
        njmax = model.njmax
        ncon = data.ncon
        contact_forces_vect = np.zeros(njmax, dtype=np.float64)
        inv_AR = np.zeros(njmax, dtype=np.float64)
        dist_njmax = np.zeros(njmax, dtype=np.float64)
        vel_njmax = np.zeros(njmax, dtype=np.float64)
        efc_address_vect = np.zeros(ncon, dtype=np.int32)
        b_vect = np.zeros(ncon, dtype=np.float64)
        d_vect = np.zeros(ncon, dtype=np.float64)
        k_vect = np.zeros(ncon, dtype=np.float64)
        aref_vect = np.zeros(njmax, dtype=np.float64)
        c_array = np.zeros(6, dtype=np.float64)
        f_normal_vect = np.zeros(ncon, dtype=np.int32)

        if ncon != 0:

            body_names = model.body_names
            combinations_vect = [i for i in combinations(body_names, 2)]
            counter_vect = np.zeros(len(combinations_vect), dtype=np.float64)
            combinations_forces_vect = np.zeros(
                len(combinations_vect), dtype=np.float64
            )
            combinations_forces_built_in = np.zeros(
                len(combinations_vect), dtype=np.float64
            )
            if self.combinations_forces_vect.shape[0] != len(combinations_forces_vect):
                self.combinations_forces_vect = np.zeros(
                    len(combinations_forces_vect), dtype=np.float64
                )
                self.combinations_forces_vect = np.transpose(
                    self.combinations_forces_vect
                )
                self.combinations_forces_built_in = np.zeros(
                    len(combinations_forces_built_in), dtype=np.float64
                )
                self.combinations_forces_buil_in = np.transpose(
                    self.combinations_forces_built_in
                )
            contact_list_tuple = [0] * njmax

            for i in range(ncon):

                # Retrieve penetration,velocity and parameters for each contact (k,b,d)
                contact = data.contact[i]
                efc_address = contact.efc_address
                efc_address_vect[i] = efc_address
                dist_njmax[efc_address] = data.active_contacts_efc_pos[efc_address]
                geom1_name = model.geom_id2name(contact.geom1)
                geom2_name = model.geom_id2name(contact.geom2)
                geom_tuple = (geom1_name, geom2_name)
                contact_list_tuple[efc_address] = geom_tuple
                index_tuple = combinations_vect.index(geom_tuple)
                counter_vect[index_tuple] += 1

                # Retrieve also the contact forces with built-in functions for comparison
                mujoco_py.functions.mj_contactForce(sim.model, sim.data, i, c_array)
                f_normal_vect[efc_address] = c_array[0]

            vel_njmax = data.efc_vel

            k_vect = sim.data.efc_KBIP[:, 0]
            b_vect = sim.data.efc_KBIP[:, 1]

            # Calc reference acceleration
            # (can be retrieved also via aref = sim.data.efc_aref)
            # aref_vect[efc_address] = -(b_vect[i]*vel_njmax[efc_address] +
            #                  k_vect[i]*dist_njmax[efc_address])  # (njmax x 1)
            for k in range(len(k_vect)):
                aref_vect[k] = -(k_vect[k] * dist_njmax[k] + b_vect[k] * vel_njmax[k])

            # Calc unconstrained acceleration
            qacc = data.qacc_unc  # unconstrained acceleration in joint space (nv x 1)
            J = data.efc_J  # constraint Jacobian (njmax x nv)
            Jqacc = J.dot(qacc)  # (njmax x nv) x (nv x 1) = (njmax x 1)

            # Unconstrained acceleration in ee space
            # Calc (aref - a0)
            # (can be retrieved also via sim.data.efc_b linear cost term: J*qacc_unc - aref)
            delta_a = Jqacc - aref_vect  # (njmax x 1)

            # Calc A + R:
            M = np.ndarray(
                shape=(len(sim.data.qvel) ** 2,), dtype=np.float64, order="C"
            )
            mujoco_py.functions.mj_fullM(sim.model, M, sim.data.qM)
            M = np.reshape(M, (len(sim.data.qvel), len(sim.data.qvel)))
            J_invM = J.dot(np.linalg.inv(M))
            A = J_invM.dot(np.transpose(J))
            max_efc = max(efc_address_vect)
            AR_mod_efc = np.zeros((max_efc + 1, max_efc + 1), dtype=np.float64)
            R = np.diag(data.efc_R)
            AR = A + R

            # Invert only the "active" part of the matrix, the rest is full of zeros and not of interest
            AR_mod_efc[0 : max_efc + 1, 0 : max_efc + 1] = AR[
                0 : max_efc + 1, 0 : max_efc + 1
            ]
            inv_AR_efc = np.linalg.pinv(AR_mod_efc)
            inv_AR = np.zeros((njmax, njmax), dtype=np.float64)
            inv_AR[0 : max_efc + 1, 0 : max_efc + 1] = inv_AR_efc[
                0 : max_efc + 1, 0 : max_efc + 1
            ]

            # Calc total contact forces vector
            f = inv_AR.dot(delta_a)
            contact_forces_vect = f

            # Sum all the contact forces with respect to the pair of bodies in contact
            # Store the values in the class to be used by plot_contact_forces() method
            # Example: body1 and body2 have 5 points of contact, the total force between the two bodies is the sum of all the contact points forces
            for ll in range(len(efc_address_vect)):
                ind_efc = efc_address_vect[ll]
                contact_tuple = contact_list_tuple[ind_efc]
                ind_tuple = combinations_vect.index(contact_tuple)
                combinations_forces_vect[ind_tuple] = (
                    combinations_forces_vect[ind_tuple] + contact_forces_vect[ind_efc]
                )
                # Do the same for forces retrieved via built-in functions
                combinations_forces_built_in[ind_tuple] = (
                    combinations_forces_built_in[ind_tuple]
                    + contact_forces_vect[ind_efc]
                )

            self.combinations_forces_vect[
                :, self.contact_forces_calls
            ] = combinations_forces_vect

            # ----------------------------------------------------------
            # Store the forces with mujoco built in method to compare
            self.combinations_forces_built_in[
                :, self.contact_forces_calls
            ] = combinations_forces_built_in

        self.contact_forces_calls += 1

    def plot_contact_forces(self):
        """Plot the contact forces happened between all the pair of bodies that came in contact during the simulation. Must be called AFTER contact_forces() method"""

        step_vect = np.arange(0, self.contact_forces_calls, 1, dtype=int)
        plt.rcParams["text.usetex"] = True

        for i in range(len(self.combinations_vect)):
            is_all_zero = np.all((self.combinations_forces_vect[i, :] == 0))
            if not is_all_zero:
                plt.figure()
                str = " and ".join(self.combinations_vect[i])
                plt.title(r"Contact force between " + str + "(Explicit Method)")
                plt.xlabel(r"Steps")
                plt.ylabel(r"Total contact force $[N]$")
                plt.grid()
                plt.plot(
                    step_vect,
                    self.combinations_forces_vect[i, 0 : self.contact_forces_calls],
                    linewidth=2,
                )

                plt.figure()
                str = " and ".join(self.combinations_vect[i])
                plt.title(r"Contact force between " + str + "(Built In Method)")
                plt.xlabel(r"Steps")
                plt.ylabel(r"Total contact force $[N]$")
                plt.grid()
                plt.plot(
                    step_vect,
                    self.combinations_forces_built_in[i, 0 : self.contact_forces_calls],
                    linewidth=2,
                )

        # TODO: solve issue where if plt.show() is called here, then the rest of the code won't execute until you close the figure
        # plt.show()

    def contact_forces_to_csv(self, sim):
        """Given a MuJoCo Sim simulation prints all the contacts parameters and forces registered into a .csv file in the current location"""

        # Check if file .csv exists or not, ask the user to keep it and append to it or replace it
        file_exists = os.path.exists("contact_data_simulation.csv")
        if (file_exists == False) and (self.ncalls == 0):
            print("---")
            print("The 📄.csv file for the contacts data doesn't exists, creating it...")
            open("contact_data_simulation.csv", "x")
            print("---")
            self.ncalls = 1

        if (file_exists == True) and (self.ncalls == 0):
            print("---")
            keep = input(
                "The 📄.csv file for the contacts data already exists, do you want to keep it? [y/n] "
            )
            if keep == "n":
                print("File 📄.csv replaced")
                os.remove("contact_data_simulation.csv")
                open("contact_data_simulation.csv", "x")
            print("---")
            self.ncalls = 1

        # Initialization of data
        data = sim.data
        model = sim.model
        ncon = data.ncon

        # Store contact info in buffer, to be appended to main file containing all the steps.
        mujoco_py.functions.mj_printData(model, data, "buffer.txt")

        # Retrieve only the lines of interest -> contact struct lines
        with open("buffer.txt", "r") as file:
            lines = file.readlines()
            index_begin_contact_mjdata = lines.index("CONTACT\n")
            index_end_contact_mjdata = lines.index("EFC_TYPE\n")
            contact_lines = lines[index_begin_contact_mjdata:index_end_contact_mjdata]
        contact_forces_string = ["contact_forces"]

        # Write on main file the info of contact structure, including the contact forces retrieved by function mj_contact_forces

        # TODO: instead of open, trasform it in: with f open etc, since this code can eventually not close the file if an error occur

        f = open("contact_data_simulation.csv", "a")
        writer = csv.writer(f)

        # TODO: optimize the nested for loops
        for i in range(len(contact_lines)):
            linesplit = contact_lines[i].split()
            if "dim" in linesplit:
                linesplit_check = contact_lines[i - 1].split()
                for k in range(ncon):
                    if str(k) in linesplit_check[0]:
                        index_contact = k
                        contact_forces_array = np.zeros(6, dtype=np.float64)

                        mujoco_py.functions.mj_contactForce(
                            model, data, index_contact, contact_forces_array
                        )

                        for kk in range(len(contact_forces_array)):
                            contact_forces_string.append(contact_forces_array[kk])
                        writer.writerow(contact_forces_string)
                        contact_forces_string = ["contact_forces"]
                        break

                writer.writerow(linesplit)
            else:
                writer.writerow(linesplit)

        f.close()

        return
