#
import csv
import os.path

import mujoco_py
import numpy as np
import pandas as pd
from click import prompt


class Printing:
    def __init__(self):
        self.ncalls = 0

    def contact_forces(self, sim):
        """Given a MuJoCo mjSim simulation,returns the vector of contact forces in explicit way, following the formula f = (A + R)^-1 * (aref - a0). Moreover, it returns the contacts parameters vectors (b,k,d)"""

        # Inizialization of vectors and data
        data = sim.data
        model = sim.model
        njmax = model.njmax
        ncon = data.ncon
        contact_forces_vect = np.zeros(njmax, dtype=np.float64)
        inv_AR = np.zeros(njmax, dtype=np.float64)
        dist_njmax = np.zeros(njmax, dtype=np.float64)
        vel_njmax = np.zeros(njmax, dtype=np.float64)
        efc_address_vect = np.zeros(ncon, dtype=np.float64)
        b_vect = np.zeros(ncon, dtype=np.float64)
        d_vect = np.zeros(ncon, dtype=np.float64)
        k_vect = np.zeros(ncon, dtype=np.float64)
        aref_vect = np.zeros(njmax, dtype=np.float64)

        if ncon != 0:

            for i in range(ncon):

                # Retrieve penetration,velocity and parameters for each contact (k,b,d)
                contact = data.contact[i]
                efc_address_vect[i] = contact.efc_address
                tau = contact.solref[0]
                damp_ratio = contact.solref[1]
                d = contact.solimp[0]
                d_vect[i] = d
                b_vect[i] = 2 / (tau * d)
                k_vect[i] = d / (
                    (tau**2) * (damp_ratio**2) * (d**2)
                )  # (njmax x 1)
                dist_njmax[i] = data.active_contacts_efc_pos[i]  # (njmax x 1)
                vel_njmax[i] = data.efc_vel[i]  # (njmax x 1)

                # Calc reference acceleration
                # (can be retrieved also via aref = sim.data.efc_aref)
                aref_vect[i] = -(
                    b_vect[i] * vel_njmax[i] + k_vect[i] * dist_njmax[i]
                )  # (njmax x 1)

            # Calc unconstrained acceleration
            qacc = data.qacc_unc  # unconstrained acceleration in joint space (nv x 1)
            J = data.efc_J  # constraint Jacobian (njmax x nv)
            Jqacc = J.dot(qacc)  # (njmax x nv) x (nv x 1) = (njmax x 1)
            # unconstrained acceleration in ee space

            # Calc (aref - a0)
            # (can be retrieved also via sim.data.efc_b[0] linear cost term: J*qacc_unc - aref)
            delta_a = -(Jqacc - aref_vect)  # (njmax x 1)

            # Calc A and R:
            # R
            R_diag = data.efc_R  # inverse constraint mass (njmax x 1)
            R = np.diag(R_diag)
            # A
            diag_inv_M = data.qLDiagInv
            inv_M = np.diag(diag_inv_M)
            J_invM = J.dot(inv_M)
            A = J_invM.dot(np.transpose(J))

            # Calc F = (A+R)^-1 * (aref - a0)
            inv_AR = np.linalg.pinv(A + R)
            contact_forces_vect = inv_AR.dot(delta_a)  # (njmax x 1)

        return [contact_forces_vect, b_vect, k_vect, d_vect, ncon, efc_address_vect]

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
        f = open("contact_data_simulation.csv", "a")
        writer = csv.writer(f)

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
