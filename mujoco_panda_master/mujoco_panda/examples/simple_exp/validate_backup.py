#
import csv
import math
import os.path
from itertools import combinations

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
        efc_address_vect = np.zeros(ncon, dtype=np.int32)
        b_vect = np.zeros(ncon, dtype=np.float64)
        d_vect = np.zeros(ncon, dtype=np.float64)
        k_vect = np.zeros(ncon, dtype=np.float64)
        aref_vect = np.zeros(njmax, dtype=np.float64)
        dist_njmax2 = np.zeros(njmax, dtype=np.float64)

        if ncon != 0:

            body_names = model.body_names
            combinations_vect = [i for i in combinations(body_names, 2)]
            counter_vect = np.zeros(len(combinations_vect), dtype=np.float64)
            combinations_forces_vect = np.zeros(
                len(combinations_vect), dtype=np.float64
            )
            contact_list_tuple = [0] * njmax

            # index()
            # sim.model.geom_id2name(contact.geom1)
            for i in range(ncon):

                # Retrieve penetration,velocity and parameters for each contact (k,b,d)
                J = data.efc_J
                contact = data.contact[i]
                efc_address = contact.efc_address
                efc_address_vect[i] = efc_address
                tau = contact.solref[0]
                damp_ratio = contact.solref[1]
                d = contact.solimp[0]
                d_vect[i] = d
                b_vect[i] = 2 / (tau * d)
                k_vect[i] = d / (
                    (tau**2) * (damp_ratio**2) * (d**2)
                )  # (njmax x 1)

                dist_njmax[efc_address] = data.active_contacts_efc_pos[efc_address]
                # Calc reference acceleration
                # (can be retrieved also via aref = sim.data.efc_aref)
                # aref_vect[efc_address] = -(b_vect[i]*vel_njmax[efc_address] +
                #                  k_vect[i]*dist_njmax[efc_address])  # (njmax x 1)
                geom1_name = model.geom_id2name(contact.geom1)
                geom2_name = model.geom_id2name(contact.geom2)
                geom_tuple = (geom1_name, geom2_name)
                contact_list_tuple[efc_address] = geom_tuple
                index_tuple = combinations_vect.index(geom_tuple)
                counter_vect[index_tuple] += 1

            # dist_njmax = J.dot(data.qpos)  # (njmax x 1)
            # mujoco_py.functions.mj_mulJacVec(model, data, dist_njmax, data.qpos);

            # vel_njmax = J.dot(data.qvel)
            mujoco_py.functions.mj_mulJacVec(model, data, vel_njmax, data.qvel)

            k_vect = sim.data.efc_KBIP[:, 0]
            b_vect = sim.data.efc_KBIP[:, 1]
            for k in range(len(k_vect)):
                aref_vect[k] = -(k_vect[k] * dist_njmax[k] + b_vect[k] * vel_njmax[k])

            # Calc unconstrained acceleration
            qacc = data.qacc_unc  # unconstrained acceleration in joint space (nv x 1)
            J = data.efc_J  # constraint Jacobian (njmax x nv)
            Jqacc = J.dot(qacc)  # (njmax x nv) x (nv x 1) = (njmax x 1)
            # unconstrained acceleration in ee space

            # Calc (aref - a0)
            # (can be retrieved also via sim.data.efc_b[0] linear cost term: J*qacc_unc - aref)
            delta_a = Jqacc - aref_vect  # (njmax x 1)
            # delta_a = sim.data.efc_b

            # Calc A and R:
            # R
            R_diag = data.efc_R  # inverse constraint mass (njmax x 1)
            # R = np.diag(R_diag)
            # A
            diag_inv_M = data.qLDiagInv
            inv_M = np.diag(diag_inv_M)
            J_invM = J.dot(inv_M)
            A = J_invM.dot(np.transpose(J))
            A_diag = np.diag(A)
            A_diag.flags.writeable = True
            M = data.qM

            # mujoco_py.functions.mj_fullM(model, M, data.qM)

            mass_matrix = np.ndarray(
                shape=(len(sim.data.qvel) ** 2,), dtype=np.float64, order="C"
            )
            mujoco_py.functions.mj_fullM(sim.model, mass_matrix, sim.data.qM)
            mass_matrix = np.reshape(
                mass_matrix, (len(sim.data.qvel), len(sim.data.qvel))
            )
            j_invM = J.dot(np.linalg.inv(mass_matrix))
            a = j_invM.dot(np.transpose(J))
            r = np.diag(data.efc_R)
            ar = a + r
            inv_ar = np.linalg.pinv(ar)
            f = inv_ar.dot(delta_a)

            # A_diag = data.efc_diagApprox
            # A = np.diag(A_diag)
            AR_diag = np.zeros(njmax, dtype=np.float64)

            for ll in range(len(efc_address_vect)):
                tuple1 = contact_list_tuple[efc_address_vect[ll]]
                index1 = combinations_vect.index(tuple1)
                temp = A_diag[efc_address_vect[ll]] * (counter_vect[index1])
                A_diag[efc_address_vect[ll]] = temp
                AR_diag[efc_address_vect[ll]] = (
                    A_diag[efc_address_vect[ll]] + R_diag[efc_address_vect[ll]]
                )
                contact_forces_vect[efc_address_vect[ll]] = (
                    1 / (AR_diag[efc_address_vect[ll]]) * delta_a[efc_address_vect[ll]]
                )
                combinations_forces_vect[index1] = (
                    combinations_forces_vect[index1]
                    + contact_forces_vect[efc_address_vect[ll]]
                )

            # AR_diag = A_diag + R_diag

            # for kk in range(len(AR_diag)):
            #     contact_forces_vect[kk] = 1/(AR_diag[kk]) * delta_a[kk]

            # Calc F = (A+R)^-1 * (aref - a0)
            # inv_AR = np.linalg.pinv(A+R)
            # contact_forces_vect = -inv_AR.dot(delta_a)  # (njmax x 1)
            # index = combinations_vect.index()

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
