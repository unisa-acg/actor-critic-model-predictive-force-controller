import random

import numpy as np

"""Functions to randomize the input of the class TrajectoryGenerator, following the
range given by the user.
The output parameters allow the creation of random 2d trajectory and a trend of
reference forces
"""


def new_waypoint_gen(bl, tr):
    """Generates a new random tuple corrispondent to a point `(x,y)` inside the
    operating area
    Args:
        bl (tuple): bottom left limit point of the operating space
        tr (tuple): top right limit point of the operating space
    """
    offset = 0.05
    x = random.uniform(bl[0] + offset, tr[0] - offset)
    y = random.uniform(bl[1] + offset, tr[1] - offset)
    return (x, y)


def _reduce_max_ampl_sine(max_ampl_user, bl, tr, xi, xf):
    """Modidifies the maximum amplitude of the sine wave, given by the user, to contain
    it in the operating zone
    Args:
        max_ampl_user (float): maximum amplitude of the sine wave given by the user
        bl (tuple): bottom left limit point of the operating space
        tr (tuple): top right limit point of the operating space
        xi (tuple): initial point
        xf (tuple): final point

    Returns:
        max_ampl_mod: reduce sine amplitude
    """

    m = (xi[1] - xf[1]) / (xi[0] - xf[0])
    if xf[0] - xi[0] < 0:
        theta = np.arctan(m) + np.pi
    else:
        theta = np.arctan(m)
    theta += np.pi / 2

    # Circumscribe the sine wave into a rectangle to make it easier to check relative
    # distances to border of operating area
    p1_rectx = xi[0] + max_ampl_user / 2 * np.cos(theta)
    p1_recty = xi[1] + max_ampl_user / 2 * np.sin(theta)

    p3_rectx = xi[0] - max_ampl_user / 2 * np.cos(theta)
    p3_recty = xi[1] - max_ampl_user / 2 * np.sin(theta)

    p2_rectx = xf[0] + max_ampl_user / 2 * np.cos(theta)
    p2_recty = xf[1] + max_ampl_user / 2 * np.sin(theta)

    p4_rectx = xf[0] - max_ampl_user / 2 * np.cos(theta)
    p4_recty = xf[1] - max_ampl_user / 2 * np.sin(theta)

    # Check if the sine-rectangle exceeds the limits of the operating area
    check_x_dir = (
        (p1_rectx > bl[0] and p1_rectx < tr[0])
        and (p2_rectx > bl[0] and p2_rectx < tr[0])
        and (p3_rectx > bl[0] and p3_rectx < tr[0])
        and (p4_rectx > bl[0] and p4_rectx < tr[0])
    )
    check_y_dir = (
        (p1_recty > bl[1] and p1_recty < tr[1])
        and (p2_recty > bl[1] and p2_recty < tr[1])
        and (p3_recty > bl[1] and p3_recty < tr[1])
        and (p4_recty > bl[1] and p4_recty < tr[1])
    )
    i = 0
    while ((check_x_dir and check_y_dir) is not True) and (i < 300):
        max_ampl_user = max_ampl_user - 0.01
        p1_rectx = xi[0] + max_ampl_user / 2 * np.cos(theta)
        p1_recty = xi[1] + max_ampl_user / 2 * np.sin(theta)

        p3_rectx = xi[0] - max_ampl_user / 2 * np.cos(theta)
        p3_recty = xi[1] - max_ampl_user / 2 * np.sin(theta)

        p2_rectx = xf[0] + max_ampl_user / 2 * np.cos(theta)
        p2_recty = xf[1] + max_ampl_user / 2 * np.sin(theta)

        p4_rectx = xf[0] - max_ampl_user / 2 * np.cos(theta)
        p4_recty = xf[1] - max_ampl_user / 2 * np.sin(theta)
        check_x_dir = (
            (p1_rectx > bl[0] and p1_rectx < tr[0])
            and (p2_rectx > bl[0] and p2_rectx < tr[0])
            and (p3_rectx > bl[0] and p3_rectx < tr[0])
            and (p4_rectx > bl[0] and p4_rectx < tr[0])
        )
        check_y_dir = (
            (p1_recty > bl[1] and p1_recty < tr[1])
            and (p2_recty > bl[1] and p2_recty < tr[1])
            and (p3_recty > bl[1] and p3_recty < tr[1])
            and (p4_recty > bl[1] and p4_recty < tr[1])
        )
        i += 1

    max_ampl_mod = max_ampl_user

    return max_ampl_mod


def _reduce_max_ampl_sine_force(max_ampl_f, max_f_ref, min_f_ref, f_start, f_end):
    """Modidifies the maximum amplitude of the sine wave for the reference force, to
    contain it in the force range
    Args:
        max_ampl_f (float): maximum amplitude of the sine wave for the reference force
        max_f_ref (float): maximum value of the reference forces
        min_f_ref (float): manimum value of the reference forces
        f_start (float): initial force value
        f_end (float): final force value

    Returns:
        max_ampl_mod: reduce sine amplitude
    """

    # Circumscribe the sine wave into a rectangle to make it easier to check relative
    # distances to border of operating area
    p1 = f_start + max_ampl_f / 2
    p2 = f_start - max_ampl_f / 2
    p3 = f_end - max_ampl_f / 2
    p4 = f_end + max_ampl_f / 2

    # Check if the sine-rectangle exceeds the limits of the operating area
    check = (
        (p1 < max_f_ref) and (p2 > min_f_ref) and (p3 > min_f_ref) and (p4 < max_f_ref)
    )
    i = 0
    while (check is not True) and (i < 300):
        max_ampl_f = max_ampl_f - 0.1
        p1 = f_start + max_ampl_f / 2
        p2 = f_start - max_ampl_f / 2
        p3 = f_end - max_ampl_f / 2
        p4 = f_end + max_ampl_f / 2

        check = (
            (p1 < max_f_ref)
            and (p2 > min_f_ref)
            and (p3 > min_f_ref)
            and (p4 < max_f_ref)
        )

        i += 1

    max_ampl_mod_f = max_ampl_f

    return max_ampl_mod_f


def _reduce_max_radius(max_radius_user, bl, tr, xi):
    """Modifies the maximum radius of the circle, given by the user, to contain it in
    operating zone
    Args:
        max_radius_user (float): maximum radius of the circle given by the user
        bl (tuple): bottom left limit point of the operating space
        tr (tuple): top right limit point of the operating space
        xi (tuple): initial point of the circle
    """
    check_x = (xi[0] - 2 * max_radius_user) > bl[0]
    check_y = ((xi[1] - max_radius_user) > bl[1]) and (
        (xi[1] + max_radius_user) < tr[1]
    )
    i = 0
    while ((check_x and check_y) is not True) and (i < 300):
        max_radius_user = max_radius_user - 0.01
        check_x = (xi[0] - 2 * max_radius_user) > bl[0]
        check_y = ((xi[1] - max_radius_user) > bl[1]) and (
            (xi[1] + max_radius_user) < tr[1]
        )
        i += 1

    max_radius_mod = max_radius_user
    return max_radius_mod


def traj_randomizer(params_randomizer_dict):
    """Generates a random set of input parameters (such as trajectory types, waypoints,
    force reference types and values) for a trajectory generator

    Args:
        params_randomizer_dict: dictionary containing the limits, operating zones,
        the maximum and minimum values of different parameters for the randomizer

    Returns:
        [waypoints, traj_timestamps, traj_types, traj_params, force_reference_types,
        force_reference_params]: returns all the necessary information and parameters
        for the trajectory generator
    """
    starting_point = params_randomizer_dict["starting_point"]  # starting_point
    operating_zone_points = params_randomizer_dict[
        "operating_zone_points"
    ]  # operating zone bottom left point and top right point
    # (i.e. [(x_bl, y_bl), (x_tr, y_tr)])
    max_n_subtraj = params_randomizer_dict[
        "max_n_subtraj"
    ]  # max number of subtrajectory
    max_vel = params_randomizer_dict[
        "max_vel"
    ]  # max velocity of the end-effector to maintain contact
    # minimum radius of the circle
    min_radius = params_randomizer_dict["min_radius"]
    # maximum radius of the circle
    max_radius = params_randomizer_dict["max_radius"]
    # minimum amplitude of the sine wave
    max_ampl = params_randomizer_dict["max_ampl"]
    # minimum frequency of the sine wave
    max_freq = params_randomizer_dict["max_freq"]
    max_f_ref = params_randomizer_dict[
        "max_f_ref"
    ]  # maximum value of the reference forces
    min_f_ref = params_randomizer_dict[
        "min_f_ref"
    ]  # minimum value of the reference forces
    # minimum amplitude of the sine
    max_ampl_f = params_randomizer_dict["max_ampl_f"]
    # wave for the force reference
    # minimum frequency of the sine
    max_freq_f = params_randomizer_dict["max_freq_f"]
    # wave for the force reference

    rand = np.random.randint(1, max_n_subtraj + 1)

    waypoints = [starting_point]
    traj_types = [None] * rand
    traj_params = [None] * rand
    traj_timestamps = [None] * rand
    force_reference_types = [None] * rand
    force_reference_params = [None] * rand

    traj_types_avail = ["line", "circle", "sine_curve"]
    force_reference_types_avail = ["cnst", "ramp", "sine_curve"]

    # Limits for traj generation
    bl = (operating_zone_points[0][1], operating_zone_points[0][0])
    tr = (operating_zone_points[1][1], operating_zone_points[1][0])

    for i in range(rand):
        # Trajectory section

        new_subtraj = traj_types_avail[np.random.randint(len(traj_types_avail))]

        # To avoid two subsequent circles or sine waves
        if (i != 0) and (
            (traj_types[i - 1] == "sine_curve") or (traj_types[i - 1] == "circle")
        ):
            new_subtraj = "line"

        # Choose the new waypoints
        if new_subtraj == "circle":
            waypoints.append(waypoints[-1])
        else:
            waypoints.append(new_waypoint_gen(bl, tr))

        traj_types[i] = new_subtraj

        xi = waypoints[i]
        xf = waypoints[i + 1]

        # Check what random type of trajectory has been chosen and generate it
        if new_subtraj == "line":
            traj_params[i] = None
            length = np.sqrt((xf[0] - xi[0]) ** 2 + (xf[1] - xi[1]) ** 2)
        elif new_subtraj == "circle":
            max_radius_mod = _reduce_max_radius(max_radius, bl, tr, xi)
            radius = random.uniform(min_radius, max_radius_mod)
            traj_params[i] = radius
            length = 2 * np.pi * radius
        elif new_subtraj == "sine_curve":
            max_ampl_mod = _reduce_max_ampl_sine(max_ampl, bl, tr, xi, xf)
            ampl = np.random.random() * max_ampl_mod
            freq = np.random.randint(1, max_freq)
            traj_params[i] = [ampl, freq]
            lin_length = np.sqrt((xf[0] - xi[0]) ** 2 + (xf[1] - xi[1]) ** 2)
            length = (lin_length / (freq * 4) + ampl) * (freq * 4 * 1.7)
        min_time_subtraj = length / max_vel

        # Timestamps section, randomly choose a duration between the minimun and the
        # maximus admissible
        if i == 0:
            traj_timestamps[i] = np.random.randint(
                min_time_subtraj, min_time_subtraj + 5
            )
        else:
            traj_timestamps[i] = traj_timestamps[i - 1] + np.random.randint(
                min_time_subtraj, min_time_subtraj + 5
            )

        # Force section
        new_subforce = force_reference_types_avail[
            np.random.randint(len(force_reference_types_avail))
        ]

        force_reference_types[i] = new_subforce

        # Check the type of force reference chosen and generate it
        if i == 0:
            if new_subforce == "cnst":
                force_reference_params[i] = np.random.randint(min_f_ref, max_f_ref)

            elif new_subforce == "ramp":
                f_start = np.random.randint(min_f_ref, max_f_ref)
                f_end = np.random.randint(min_f_ref, max_f_ref)
                force_reference_params[i] = [f_start, f_end]

            elif new_subforce == "sine_curve":
                f_start = np.random.randint(min_f_ref, max_f_ref)
                f_end = np.random.randint(min_f_ref, max_f_ref)

                max_ampl_mod_f = _reduce_max_ampl_sine_force(
                    max_ampl_f, max_f_ref, min_f_ref, f_start, f_end
                )
                ampl_f = np.random.random() * max_ampl_mod_f
                freq_f = np.random.randint(1, max_freq_f)
                force_reference_params[i] = [f_start, f_end, ampl_f, freq_f]

        else:
            if force_reference_types[i - 1] == "cnst":
                prev_f_value = force_reference_params[i - 1]

            elif (
                force_reference_types[i - 1] == "ramp"
                or force_reference_types[i - 1] == "sine_curve"
            ):
                prev_f_value = force_reference_params[i - 1][1]

            # Keep the ramp start force in an interval (-30,+30) with respect to the
            # previous ending reference force
            around = 1
            low_f_value = np.maximum(min_f_ref, prev_f_value - around)
            high_f_value = np.minimum(max_f_ref, prev_f_value + around)

            if new_subforce == "cnst":
                force_reference_params[i] = np.random.randint(low_f_value, high_f_value)

            elif new_subforce == "ramp":
                f_start = np.random.randint(low_f_value, high_f_value)
                f_end = np.random.randint(min_f_ref, max_f_ref)
                force_reference_params[i] = [f_start, f_end]

            elif new_subforce == "sine_curve":
                f_start = np.random.randint(low_f_value, high_f_value)
                f_end = np.random.randint(min_f_ref, max_f_ref)

                max_ampl_mod_f = _reduce_max_ampl_sine_force(
                    max_ampl_f, max_f_ref, min_f_ref, f_start, f_end
                )
                ampl_f = np.random.random() * max_ampl_mod_f
                freq_f = np.random.randint(1, max_freq_f)
                force_reference_params[i] = [f_start, f_end, ampl_f, freq_f]

    # For dimension equality insert the zero at the beginning of the timestamps vector
    traj_timestamps.insert(0, 0)

    return [
        waypoints,
        traj_timestamps,
        traj_types,
        traj_params,
        force_reference_types,
        force_reference_params,
    ]
