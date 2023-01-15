class ForceController():

    def __init__(self, k_f, ki_f):
        self.k_f = k_f
        self.ki_f = ki_f
        self.sum_force_error = 0

    def force_controller(self, f_d, f_e, dt):
        force_error = f_e - f_d
        self.sum_force_error += force_error * dt

        x_f = force_error * self.k_f + self.sum_force_error * self.ki_f

        return x_f