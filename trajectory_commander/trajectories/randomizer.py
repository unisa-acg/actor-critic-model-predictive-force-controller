import random
import numpy as np

"""Functions to randomize the input of the class TrajectoryGenerator, following the range given by the user.
The output parameters allow the creation of random 2d trajectory and a trend of reference forces
"""

def new_waypoint_gen(bl, tr):
    """It generates a new random tuple corrispondent to a point in the operating space
    Args:
        bl: the bottom left point of the opearting space
        tr: the top right point of the opearting space
    """
    offset = 0.05
    x = random.uniform(bl[0] + offset, tr[0] - offset)
    y = random.uniform(bl[1] + offset, tr[1] - offset)
    return (x, y)


def reduce_max_ampl_sine(max_ampl_user, bl, tr, xi, xf):
    """It modifies the maximum amplitude of the sine wave, given by the user, to contain it in operating zone
    Args:
        max_ampl_user: maximum amplitude of the sine wave given by the user
        bl: the bottom left point of the opearting space
        tr: the top right point of the opearting space
        xi: initial point
        xf: final point
    """
    # 
    m = (xi[1] - xf[1]) / (xi[0] - xf[0])
    if ((xf[0] - xi[0] < 0)):
        theta = np.arctan(m) + np.pi
    else:
        theta = np.arctan(m)
    theta += np.pi / 2

    p1_rectx = xi[0] + max_ampl_user / 2 * np.cos(theta)
    p1_recty = xi[1] + max_ampl_user / 2 * np.sin(theta)

    p3_rectx = xi[0] - max_ampl_user / 2 * np.cos(theta)
    p3_recty = xi[1] - max_ampl_user / 2 * np.sin(theta)

    p2_rectx = xf[0] + max_ampl_user / 2 * np.cos(theta)
    p2_recty = xf[1] + max_ampl_user / 2 * np.sin(theta)

    p4_rectx = xf[0] - max_ampl_user / 2 * np.cos(theta)
    p4_recty = xf[1] - max_ampl_user / 2 * np.sin(theta)

    check_x_dir = (p1_rectx > bl[0] and p1_rectx < tr[0]) and (
        p2_rectx > bl[0] and p2_rectx < tr[0]) and (
            p3_rectx > bl[0] and p3_rectx < tr[0]) and (p4_rectx > bl[0]
                                                        and p4_rectx < tr[0])
    check_y_dir = (p1_recty > bl[1] and p1_recty < tr[1]) and (
        p2_recty > bl[1] and p2_recty < tr[1]) and (
            p3_recty > bl[1] and p3_recty < tr[1]) and (p4_recty > bl[1]
                                                        and p4_recty < tr[1])
    i = 0
    while ((check_x_dir and check_y_dir) != True):
        max_ampl_user = max_ampl_user - 0.01
        p1_rectx = xi[0] + max_ampl_user / 2 * np.cos(theta)
        p1_recty = xi[1] + max_ampl_user / 2 * np.sin(theta)

        p3_rectx = xi[0] - max_ampl_user / 2 * np.cos(theta)
        p3_recty = xi[1] - max_ampl_user / 2 * np.sin(theta)

        p2_rectx = xf[0] + max_ampl_user / 2 * np.cos(theta)
        p2_recty = xf[1] + max_ampl_user / 2 * np.sin(theta)

        p4_rectx = xf[0] - max_ampl_user / 2 * np.cos(theta)
        p4_recty = xf[1] - max_ampl_user / 2 * np.sin(theta)
        check_x_dir = (p1_rectx > bl[0] and p1_rectx < tr[0]) and (
            p2_rectx > bl[0] and
            p2_rectx < tr[0]) and (p3_rectx > bl[0] and p3_rectx < tr[0]) and (
                p4_rectx > bl[0] and p4_rectx < tr[0])
        check_y_dir = (p1_recty > bl[1] and p1_recty < tr[1]) and (
            p2_recty > bl[1] and
            p2_recty < tr[1]) and (p3_recty > bl[1] and p3_recty < tr[1]) and (
                p4_recty > bl[1] and p4_recty < tr[1])
        i += 1
        if i > 300:
            break

    max_ampl_mod = max_ampl_user

    return max_ampl_mod

def reduce_max_radius(max_radius_user, bl, tr, xi):
    """It modifies the maximum radius of the circle, given by the user, to contain it in operating zone
    Args:
        max_radius_user: maximum radius of the circle given by the user
        bl: the bottom left point of the opearting space
        tr: the top right point of the opearting space
        xi: initial point
    """
    check_x = ((xi[0] - 2 * max_radius_user) > bl[0])
    check_y = ((xi[1] - max_radius_user) > bl[1]) and (
        (xi[1] + max_radius_user) < tr[1])
    i = 0
    while ((check_x and check_y) != True):
        max_radius_user = max_radius_user - 0.01
        check_x = ((xi[0] - 2 * max_radius_user) > bl[0])
        check_y = ((xi[1] - max_radius_user) > bl[1]) and (
            (xi[1] + max_radius_user) < tr[1])
        if i > 300:
            break

    max_radius_mod = max_radius_user
    return max_radius_mod

def traj_randomizer(params_randomizer):
    """It modifies the maximum radius of the circle, given by the user, to contain it in operating zone
    Args:
        params_randomizer: dictionary containing the limits, operating zones and the maximum and minimum values of different parameters for the randomizer
    """
    starting_point = params_randomizer['starting_point']                    #starting_point
    operating_zone_points = params_randomizer['operating_zone_points']      #operating zone bottom left point and top right point (i.e. [(x_bl, y_bl), (x_tr, y_tr)])
    max_n_subtraj = params_randomizer['max_n_subtraj']                      #max number of subtrajectory
    max_vel = params_randomizer['max_vel']                                  #max velocity of the end-effector to maintain contact 
    min_radius = params_randomizer['min_radius']                            #minimum radius of the circle
    max_radius = params_randomizer['max_radius']                            #maximum radius of the circle
    max_ampl = params_randomizer['max_ampl']                                #minimum amplitude of the sine wave
    max_freq = params_randomizer['max_freq']                                #minimum amplitude of the sine wave
    max_f_ref = params_randomizer['max_f_ref']                              #maximum value of the reference forces
    min_f_ref = params_randomizer['min_f_ref']                              #minimum value of the reference forces

    rand = np.random.randint(1, max_n_subtraj + 1)

    waypoints = [starting_point]
    traj_types = [None] * rand
    traj_params = [None] * rand
    traj_timestamps = [None] * rand
    force_reference_types = [None] * rand
    force_reference_params = [None] * rand

    traj_types_avail = ['line', 'circle', 'sine_curve']
    force_reference_types_avail = ['cnst', 'ramp']

    #limits for traj generation
    bl = (operating_zone_points[0][1], operating_zone_points[0][0])
    tr = (operating_zone_points[1][1], operating_zone_points[1][0])

    for i in range(rand):
        # traj section
        new_subtraj = traj_types_avail[np.random.randint(
            len(traj_types_avail))]

        if (i != 0) and ((traj_types[i - 1] == 'sine_curve') or
                         (traj_types[i - 1] == 'circle')):
            new_subtraj = 'line'

        if new_subtraj == 'circle':
            waypoints.append(waypoints[-1])
        else:
            waypoints.append(new_waypoint_gen(bl, tr))

        traj_types[i] = new_subtraj

        xi = waypoints[i]
        xf = waypoints[i + 1]
        if new_subtraj == 'line':
            traj_params[i] = None
            length = np.sqrt((xf[0] - xi[0])**2 + (xf[1] - xi[1])**2)
        elif new_subtraj == 'circle':
            max_radius_mod = reduce_max_radius(max_radius, bl, tr, xi)
            radius = random.uniform(min_radius, max_radius_mod)
            traj_params[i] = radius
            length = 2 * np.pi * radius
        elif new_subtraj == 'sine_curve':
            max_ampl_mod = reduce_max_ampl_sine(max_ampl, bl, tr, xi, xf)
            ampl = np.random.random() * max_ampl_mod
            freq = np.random.randint(1, max_freq)
            traj_params[i] = [ampl, freq]
            lin_length = np.sqrt((xf[0] - xi[0])**2 + (xf[1] - xi[1])**2)
            length = (lin_length / (freq * 4) + ampl) * (freq * 4 * 1.7)
        min_time_subtraj = length / max_vel

        # time section
        if i == 0:
            traj_timestamps[i] = np.random.randint(min_time_subtraj,
                                                   min_time_subtraj + 5)
        else:
            traj_timestamps[i] = traj_timestamps[i - 1] + np.random.randint(
                min_time_subtraj, min_time_subtraj + 5)

        # force section
        new_subforce = force_reference_types_avail[np.random.randint(
            len(force_reference_types_avail))]
        force_reference_types[i] = new_subforce

        if i == 0:
            if new_subforce == 'cnst':
                force_reference_params[i] = np.random.randint(min_f_ref, max_f_ref)

            elif new_subforce == 'ramp':
                f_start = np.random.randint(min_f_ref, max_f_ref)
                f_end = np.random.randint(min_f_ref, max_f_ref)
                force_reference_params[i] = [f_start, f_end]
        else:
            if force_reference_types[i-1] == 'cnst':
                prev_f_value  = force_reference_params[i-1]
                
            elif force_reference_types[i-1] == 'ramp':
                prev_f_value  = force_reference_params[i-1][1]

            around = 30
            low_f_value = np.maximum(min_f_ref, prev_f_value - around)
            high_f_value = np.minimum(max_f_ref, prev_f_value + around)
  
            if new_subforce == 'cnst':             
                force_reference_params[i] = np.random.randint(low_f_value, high_f_value)

            elif new_subforce == 'ramp':
                f_start = np.random.randint(low_f_value, high_f_value)
                f_end = np.random.randint(min_f_ref, max_f_ref)
                force_reference_params[i] = [f_start, f_end]

    traj_timestamps.insert(0, 0)

    return waypoints, traj_timestamps, traj_types, traj_params, force_reference_types, force_reference_params