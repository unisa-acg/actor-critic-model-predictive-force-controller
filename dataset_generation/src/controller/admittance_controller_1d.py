import xdrlib
import numpy as np


class AdmittanceController1D():
    """
    It represents a unidimensional position controller with a force-feedback loop. 
    Assume that every variable is within end-effector frame, a.k.a.
    a detached end-effector with a moving reference frame.
    """

    def __init__(self, M_d_inv, K_P, K_D, K_F):
        """
        Parameters
        ----------
        M_d_inv : np.matrix
            1d-array with the Intertia matrix
        K_P : np.matrix
            1d-array with the spring constant
        K_D : np.matrix
            1d-array with the damping constant    
        K_F : np.matrix
            1d-array with the force gain  
        """
        self.M_d_inv = M_d_inv
        self.K_P = K_P
        self.K_D = K_D
        self.K_F = K_F
        self.reset()

    def set_reference(self, reference_pos, reference_force):
        """
        Set a position followed by a force reference.

        Parameters
        ----------
        reference : np.ndarray
            2d-array of form [x_d, f_d]
        """
        self.x_d = reference_pos
        self.f_d = reference_force

    def get_reference(self):
        """
        Get the reference of the controller.

        Returns
        -------
        np.ndarray
            2d-array of form [x_d, f_d]
        """
        return np.array([self.x_d, self.f_d])

    def update(self, force, feedback: np.ndarray, dt: np.float64):
        """
        Receives a feedback comprised by the force felt by a force sensor.

        Parameters
        ----------
        feedback : np.ndarray
            1d-array with the force read by a force sensor
        dt : np.float64
            Current duration of control loop in seconds.

        Returns
        -------
        x_c : np.ndarray
            1d-array with the controlled variable
        """
        # The force exerted by the robot is the opposite of the force read by the sensor
        self.f_e = force
        self.x_c, self.x_c_dot = feedback

        delta_f = self.K_F * (self.f_e - self.f_d)
        delta_pos = self.K_P * (self.x_c - self.x_d)
        delta_vel = self.K_D * self.x_c_dot

        # Internal states
        sum = delta_f - delta_pos - delta_vel
        self.x_dot_dot_c = self.M_d_inv * sum
        self.x_dot_c += self.x_dot_dot_c * dt
        self.x_c += self.x_dot_c * dt

        # Returns the controlled variable
        return self.x_c

    def reset(self):
        self.x_dot_c = 0
