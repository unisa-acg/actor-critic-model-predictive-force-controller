# ACMPFC

This simulation consists in an environment composed of a Franka Emika Panda robot, with attached at the end-effector a sphere, coming in contact with a fixed plane.
The robot is controlled via an impedance controller that activates the force control when in proximity of the table, following the reference force along the _z_-axis, while the movement along the _x_ and _y_-axis follow a reference position.
The force control is effectuated through makes use of a Reinforcement Learning (RL) approach.
The input position and the force reference given to the robot are obtained exploiting the utilities of [trajectories_generation_utilities](/trajectories_generation_utilities/).
These tools are used to generate an approaching trajectory and, once in contact with the table, random trajectories.
The trajectory on which you want to test the strategy must be manually defined. Currently there is an example and it is the same visible in the paper to which this code is attached.
This script requires a normalization file retrieved from the exploration phase explained in the paper. An example of it is located in folder [norm_data](../ACMPFC/norm_data).

## How to launch it

Make sure you have followed the instructions in [getting started](../../Readme.md) before launching the following instructions.
The following lines will launch the test simulation and output the result of the ACMPFC usage, together with the regularizers usage descripted in the paper.

```sh
    * Open a terminal
    roscore 
    * Open another terminal
    cd 
    cd environment_identification
    source set_env.sh
    cd ACMPFC
    python3 ACMPFC_online_controller.py
    * Open another terminal
    cd 
    cd environment_identification
    source set_env.sh
    cd ACMPFC
    python3 ACMPFC_online_controller.py
    python3 mujoco_panda_simulation.py
```

## Expected output of test ACMPFC

The simulation will generate two main outputs: eleven figures and two .csv files.
The figures represent:

1. the generated trajectory followed by the end-effector along the _x_ and _y_-axis;
2. the generated force followed by the end-effector along the _z_-axis;
3. the torques applied to the first six joints (useful for checking non-saturation);
4. the end-effector position along the directions _x_, _y_ over time and on the plane;
5. the end-effector velocity along the directions _x_, _y_ over time;
6. the contact force along time;
7. the PI controller output, the ORACLE correction and their sum along time;
8. 3d plot of the force tracking error versus x_d and xd_dot;
9. the force tracking error along time;
10. 3d plot of the prediction error versus x_d and xd_dot;
11. the prediction error along time;

These graphs provide graphical display tools that may or may not be used.

The .csv files are contained in the folder _output_, divided into two subfolders:

* _traj_generated_csv_: store the generated trajectories (one file per trajectory)
* _traj_resampled_csv_: store the resampled trajectories (one file per trajectory)
