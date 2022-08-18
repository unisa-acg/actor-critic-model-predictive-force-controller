#
import csv
import os.path
from itertools import combinations

import matplotlib.pyplot as plt
import mujoco_py
import numpy as np


class MujocoContactValidation:
    """
    Class that contains method to calculate, plot and store in a .csv the
    contact forces and info happening during the MuJoCo simulation

    Methods
    -------

    * __init__(sim,steps): instantiate the variables need for the other methods

    * contact_forces (sim): calculate the contact forces happened during the simulation
    by formula ``f = (A + R)^-1 * (aref - a0)``
            * calc_delta_a(): calculate the (aref - a0) component
            * calc_invAR(sim,efc_address_vect): calculate the (A+R)^-1 component

    * plot_contact_forces(): after calling contact_forces() method, can plot the forces
    registered

    * contact_forces_to_csv(sim): stores all the contact forces registered during the
    simulation into a .csv file in current location

    * contact_forces_to_csv_single_contact(sim)
    """

    def __init__(self, sim, steps):
        self.ncalls = 0
        self.ncallspanda = 0
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
        self.contact_forces_to_csv_call = 0
        self.forces_vect_mujocoenv = np.zeros(
            (6, steps), dtype=np.float64, order="C")

    # ---------------------------------------------------------------------------- #
    #                          Contact forces calculation                          #
    # ---------------------------------------------------------------------------- #

    # ------------------------------- Subfunctions ------------------------------- #

    def calc_invAR(self, sim, efc_address_vect):
        """Subfunction that calculates the (A+R)^-1 component

        Args:
            sim (mjSim): MuJoCo simulator istance at current timestep
            efc_address_vect (NDArray): vector that stores the indexes of the contact
                                        points

        Returns:
            NDArray: (A+R)^-1 component
        """

        M = np.ndarray(shape=(len(sim.data.qvel) ** 2,),
                       dtype=np.float64, order="C")
        mujoco_py.functions.mj_fullM(sim.model, M, sim.data.qM)
        M = np.reshape(M, (len(sim.data.qvel), len(sim.data.qvel)))
        J = sim.data.efc_J
        J_invM = J.dot(np.linalg.inv(M))
        A = J_invM.dot(np.transpose(J))
        max_efc = max(efc_address_vect)
        AR_mod_efc = np.zeros((max_efc + 1, max_efc + 1), dtype=np.float64)
        R = np.diag(sim.data.efc_R)
        AR = A + R

        # Invert only the "active" part of the matrix, the rest is full of zeros and
        # not of interest
        AR_mod_efc[0: max_efc + 1, 0: max_efc + 1] = AR[
            0: max_efc + 1, 0: max_efc + 1
        ]
        inv_AR_efc = np.linalg.pinv(AR_mod_efc)
        inv_AR = np.zeros((sim.model.njmax, sim.model.njmax), dtype=np.float64)
        inv_AR[0: max_efc + 1, 0: max_efc + 1] = inv_AR_efc[
            0: max_efc + 1, 0: max_efc + 1
        ]

        return inv_AR

    def calc_delta_a(self, sim):
        """Subfunction that calculates the (a0 - aref) component

        Args:
            sim (mjSim): MuJoCo simulator instance at current timestep

        Returns:
            NDArray: (a0 - aref) component
        """

        # ----------------------------- Data initializing ---------------------------- #
        ncon = sim.data.ncon
        model = sim.model
        data = sim.data
        njmax = model.njmax

        b_vect = np.zeros(ncon, dtype=np.float64)
        # d_vect = np.zeros(ncon, dtype=np.float64)
        k_vect = np.zeros(ncon, dtype=np.float64)
        aref_vect = np.zeros(njmax, dtype=np.float64)
        c_array = np.zeros(6, dtype=np.float64)
        dist_njmax = np.zeros(njmax, dtype=np.float64)
        vel_njmax = np.zeros(njmax, dtype=np.float64)
        efc_address_vect = np.zeros(ncon, dtype=np.int32)
        f_normal_vect = np.zeros(ncon, dtype=np.float64)

        body_names = sim.model.body_names
        combinations_vect = [i for i in combinations(body_names, 2)]
        counter_vect = np.zeros(len(combinations_vect), dtype=np.float64)
        combinations_forces_vect = np.zeros(
            len(combinations_vect), dtype=np.float64)
        combinations_forces_built_in = np.zeros(
            len(combinations_vect), dtype=np.float64
        )

        if self.combinations_forces_vect.shape[0] != len(combinations_forces_vect):
            self.combinations_forces_vect = np.zeros(
                len(combinations_forces_vect), dtype=np.float64
            )
            self.combinations_forces_vect = np.transpose(
                self.combinations_forces_vect)
            self.combinations_forces_built_in = np.zeros(
                len(combinations_forces_built_in), dtype=np.float64
            )
            self.combinations_forces_built_in = np.transpose(
                self.combinations_forces_built_in
            )
        contact_list_tuple = [0] * njmax
        # ---------------------------------------------------------------------------- #

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
            mujoco_py.functions.mj_contactForce(
                sim.model, sim.data, i, c_array)
            f_normal_vect[efc_address] = c_array[0]

        vel_njmax = sim.data.efc_vel
        k_vect = sim.data.efc_KBIP[:, 0]
        b_vect = sim.data.efc_KBIP[:, 1]

        # Calc reference acceleration --- aref ---
        for k in range(len(k_vect)):
            aref_vect[k] = -(k_vect[k] * dist_njmax[k] +
                             b_vect[k] * vel_njmax[k])

        # Calc unconstrained acceleration --- a0 ---
        # unconstrained acceleration in joint space (nv x 1)
        qacc = sim.data.qacc_unc
        J = sim.data.efc_J  # constraint Jacobian (njmax x nv)
        Jqacc = J.dot(qacc)  # (njmax x nv) x (nv x 1) = (njmax x 1)

        # Calc --- delta a ---
        delta_a = Jqacc - aref_vect  # (njmax x 1)

        return [
            delta_a,
            efc_address_vect,
            combinations_vect,
            contact_list_tuple,
            combinations_forces_vect,
            combinations_forces_built_in,
            f_normal_vect,
        ]

    # ------------------------------- Main function ------------------------------ #

    def contact_forces(self, sim):
        """Given a MuJoCo mjSim simulation, returns the vector of contact forces,
        following the formula ``f = (A + R)^-1 * (aref - a0)``.
        Moreover, it stores the value to be plotted using ``plot_contact_forces()``
        method``

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

            # Calc ---delta_a---
            [
                delta_a,
                efc_address_vect,
                combinations_vect,
                contact_list_tuple,
                combinations_forces_vect,
                combinations_forces_built_in,
                f_normal_vect,
            ] = self.calc_delta_a(sim)

            # Calc ---inv_AR---
            inv_AR = self.calc_invAR(sim, efc_address_vect)

            # Calc ---total contact forces vector---
            contact_forces_vect = -inv_AR.dot(delta_a)

            # Sum all the contact forces with respect to the pair of bodies in contact
            # Store the values in the class to be used by plot_contact_forces() method
            # Example: body1 and body2 have 5 points of contact, the total force between
            # the two bodies is the sum of all the contact points forces
            for ll in range(len(efc_address_vect)):
                ind_efc = efc_address_vect[ll]
                contact_tuple = contact_list_tuple[ind_efc]
                ind_tuple = combinations_vect.index(contact_tuple)
                combinations_forces_vect[ind_tuple] = (
                    combinations_forces_vect[ind_tuple] +
                    contact_forces_vect[ind_efc]
                )
                # Do the same for forces retrieved via built-in functions
                combinations_forces_built_in[ind_tuple] = (
                    combinations_forces_built_in[ind_tuple] +
                    f_normal_vect[ind_efc]
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

        return [
            self.combinations_forces_built_in,
            self.combinations_forces_vect,
            self.combinations_vect,
        ]

    def contact_forces_mujocoenv(self, env, geom1_name, geom2_name):
        """Substitute function to the heavy contact_forces(), which can slow down
        simulation times. It implements the built in methods to retrieve the forces even
         in presence of friction. Stores them in the class in order to plot them.

        Args:
            env: MuJoCo environment from which allows access to sim
                 (mjSim -> MuJoCo simulation instance at current timestep)
            geom1_name, geom2_name: name of the geometries whose mutual contact you want
                                    to check
        """

        sim = env.sim
        env.check_contact("sphere", "table_collision")

        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            if (
                sim.model.geom_id2name(contact.geom1) == geom1_name
                or sim.model.geom_id2name(contact.geom1) == geom2_name
            ) and (
                sim.model.geom_id2name(contact.geom2) == geom1_name
                or sim.model.geom_id2name(contact.geom2) == geom2_name
            ):
                forces_vect = np.zeros(6, dtype=np.float64)
                mujoco_py.functions.mj_contactForce(
                    sim.model, sim.data, i, forces_vect)
                self.forces_vect_mujocoenv[:,
                                           self.ncalls_force_mujocoenv] = forces_vect

        self.ncalls_force_mujocoenv += 1

    def plot_contact_forces_mujocoenv(self):
        """plot the forces directions and the total magnitude happened between the pair
        of bodies specified in ``contact_forces_mujocoenv()``.
        Must be called AFTER ``contact_forces_mujocoenv()`` method

        Returns:
            fig
        """

        fx = self.forces_vect_mujocoenv[2, 0: self.ncalls_force_mujocoenv]
        fy = self.forces_vect_mujocoenv[1, 0: self.ncalls_force_mujocoenv]
        fz = self.forces_vect_mujocoenv[0, 0: self.ncalls_force_mujocoenv]
        fxyz_sq = np.power(fx, 2) + np.power(fy, 2) + np.power(fz, 2)
        f_mag = np.power(fxyz_sq, 0.5)
        step_vect = np.arange(0, self.ncalls_force_mujocoenv, 1)
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(step_vect, fx)
        axs[0, 0].set_title("$F_x$")
        axs[0, 0].grid()
        axs[0, 1].plot(step_vect, fy, "tab:orange")
        axs[0, 1].set_title("$F_y$")
        axs[0, 1].grid()
        axs[1, 0].plot(step_vect, fz, "tab:green")
        axs[1, 0].set_title("$F_z$")
        axs[1, 0].grid()
        axs[1, 1].plot(step_vect, f_mag, "tab:red")
        axs[1, 1].set_title("$F_{tot}$")
        axs[1, 1].grid()

        for ax in axs.flat:
            ax.set(xlabel="Steps", ylabel="$[N]$")

        return axs

    # ---------------------------------------------------------------------------- #
    #              Plot contact forces (explicit and built in method)              #
    # ---------------------------------------------------------------------------- #

    def plot_contact_forces(self):
        """Plot the contact forces happened between all the pair of bodies that came in
        contact during the simulation.
        Must be called AFTER ``contact_forces()`` method"""

        step_vect = np.arange(0, self.contact_forces_calls, 1, dtype=int)
        plt.rcParams["text.usetex"] = True

        for i in range(len(self.combinations_vect)):
            is_all_zero = np.all((self.combinations_forces_vect[i, :] == 0))
            if not is_all_zero:
                plt.figure()
                str = " and ".join(self.combinations_vect[i])
                plt.title(r"Contact force between " +
                          str + "(Explicit Method)")
                plt.xlabel(r"Steps")
                plt.ylabel(r"Total contact force $[N]$")
                plt.grid()
                plt.plot(
                    step_vect,
                    self.combinations_forces_vect[i,
                                                  0: self.contact_forces_calls],
                    linewidth=2,
                )

                plt.figure()
                str = " and ".join(self.combinations_vect[i])
                plt.title(r"Contact force between " +
                          str + "(Built In Method)")
                plt.xlabel(r"Steps")
                plt.ylabel(r"Total contact force $[N]$")
                plt.grid()
                plt.plot(
                    step_vect,
                    self.combinations_forces_built_in[i,
                                                      0: self.contact_forces_calls],
                    linewidth=2,
                    color="red",
                )

    # ---------------------------------------------------------------------------- #
    #                       Save contacts informations to csv                      #
    # ---------------------------------------------------------------------------- #

    def contact_forces_to_csv(self, sim):
        """Given a MuJoCo Sim simulation prints all
        the contacts parameters and forces registered
        into a .csv file in the current location

        Args:
            sim (mjSim): MuJoCo simulator instance at current timestep
        """

        # Check if file .csv exists or not, ask the user to keep it and append to it
        # or replace it
        file_exists = os.path.exists("contact_data_simulation.csv")
        if (file_exists is False) and (self.ncalls == 0):
            print("---")
            print("The 📄.csv file for the contacts data doesn't exists, creating it.")
            open("contact_data_simulation.csv", "x")
            print("---")
            self.ncalls = 1

        if (file_exists is True) and (self.ncalls == 0):
            print("---")
            keep = input(
                "The 📄.csv file for the contacts data already exists,"
                + " do you want to keep it? [y/n] "
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
                contact_lines[0] + "Step " +
                str(self.contact_forces_to_csv_call)
            )

        contact_forces_string = ["contact_forces"]

        # Write on main file the info of contact structure, including the contact forces
        # retrieved by function mj_contact_forces

        self.contact_forces_to_csv_call += 1

        # TODO: optimize the nested for loops
        # TODO: instead of geom 1 and geom 0 transform the names
        # into the real names of the bodies

        if ncon != 0:
            f = open("contact_data_simulation.csv", "a")
            writer = csv.writer(f)

            # populate list with lines that we want to exclude from contact structure
            lines_to_exclude = ["exclude", "includemargin", "pos", "frame"]

            for i in range(len(contact_lines)):
                linesplit = contact_lines[i].split()

                # exclude the line if present in list of exclusion
                flag = any(x in linesplit for x in lines_to_exclude)
                if flag is True:
                    continue

                # add contact forces line to the structure to be written to file
                if "dim" in linesplit:
                    linesplit_check = contact_lines[i - 1].split()
                    for k in range(ncon):
                        if str(k) in linesplit_check[0]:
                            index_contact = k
                            contact_forces_array = np.zeros(
                                6, dtype=np.float64)

                            mujoco_py.functions.mj_contactForce(
                                model, data, index_contact, contact_forces_array
                            )

                            for kk in range(len(contact_forces_array)):
                                contact_forces_string.append(
                                    contact_forces_array[kk])
                            writer.writerow(contact_forces_string)
                            contact_forces_string = ["contact_forces"]

                    writer.writerow(linesplit)
                else:
                    writer.writerow(linesplit)

            f.close()

    def contact_forces_to_csv_single_contact(self, sim):
        """Given a MuJoCo Sim simulation prints the contact info (penetration, velocity
        of penetration, contact force) to a .csv file in the current location (use if
        ONLY 1 contact is registered and foreseen in the simulation!).
        Otherwise use contact_forces_to_csv() method

        Args:
            sim (mjSim): MuJoCo simulator instance at current timestep
        """

        file_exists = os.path.exists("contact_information_panda.csv")
        if (file_exists is False) and (self.ncallspanda == 0):
            f = open("contact_information_panda.csv", "x")
            writer = csv.writer(f)
            header = ["Penetration", "Velocity of deformation",
                      "Contact force", "Ncon"]
            writer.writerow(header)
            f.close()
            self.ncallspanda = 1

        if (file_exists is True) and (self.ncallspanda == 0):

            os.remove("contact_information_panda.csv")
            f = open("contact_information_panda.csv", "x")
            writer = csv.writer(f)
            header = ["Penetration", "Velocity of deformation",
                      "Contact force", "Ncon"]
            writer.writerow(header)
            f.close()
            self.ncallspanda = 1

        # Initialization of data
        data = sim.data
        model = sim.model
        ncon = data.ncon

        contact_forces_array = np.zeros(6, dtype=np.float64)
        # Store contact info in buffer, to be appended to main file containing all the
        # steps.
        if ncon == 1:
            efc_address = data.contact[0].efc_address
            dist = data.active_contacts_efc_pos[efc_address]
            vel = data.efc_vel[efc_address]
            mujoco_py.functions.mj_contactForce(
                model, data, 0, contact_forces_array)
        else:
            dist = 0
            vel = 0

        data_to_be_written = [dist, vel, contact_forces_array[0], ncon]

        with open("contact_information_panda.csv", "a") as file:
            writer = csv.writer(file)
            writer.writerow(data_to_be_written)

        return
