# Demo #1 - Sphere-plane simulation

## Simulation description

This demo consists in a sphere falling on a plane and resting on it. The aim of this simulation is to compare the forces retrieved by the _mujoco_validation_ module utilities, which exploit a reconstructed MuJoCo explicit model to calculate them, and compare the results with the built-in functions results, which are instead based on solving a convex optimization problem. Moreover, all the contact info at the current time step, such as penetration and velocity of deformation of each contact point, are appended to a .csv file.
All the parameters of the simulation, like stiffness of the contact, mass of the sphere etc. are visible inside the XML model in the demo code.


## How to launch it

Clone the repository (if not done previously) and inside the main folder execute the following commands to source the environment and launch the demo:

```sh
    source set_env.sh
    cd demos/contact_info_retrieval
    python3 demo_sphere_plane_simulation.py
```

## Expected output

The simulation will output two figures showing the contact force history during the simulation, calculated with the explicit method and the built-in one. Moreover, the results for the contact info are stored in the _contact_data_simulation.csv_ file in the current working directory.
For the default simulator parameters the output images will look like:

<!-- ![picture 2](../../images/3381406f5781268c9682f4df48b43fd5a28be1c939bd4287c74a1e59cfbe440e.png)
![picture 3](../../images/2021ca2a09129229e0b4c02a2f2e2ae5815694b53dbd8eeb7aef8e76b6948e97.png)   -->

Built-in Method             |  Explicit Method
:-------------------------:|:-------------------------:
![pic1](../../images/2021ca2a09129229e0b4c02a2f2e2ae5815694b53dbd8eeb7aef8e76b6948e97.png)  |  ![pic2](../../images/3381406f5781268c9682f4df48b43fd5a28be1c939bd4287c74a1e59cfbe440e.png)


# Demo #2 - Robot-plane simulation

## Simulation description

This simulation consists in an environment composed of a Franka Emika Panda robot, with attached at the end-effector a sphere, coming in contact with a fixed plane. The robot is controlled via a hybrid force/motion controller that activates the force control when in proximity of the table. It tries to stabilize the force between the two bodies at $50$ $N$. As the previous simulation, all the infos about the forces acting in the contacts are plotted and stored, with all the other contact info at current step, in a .csv file.

## How to launch it

Clone the repository (if not done previously) and inside the main folder execute the following commands to source the environment and launch the demo:

```sh
    source set_env.sh
    cd demos/contact_info_retrieval
    python3 demo_panda_sphere_simulation.py
```

## Expected output

The simulation will show the robot coming in contact and stabiling it, then the figures of the forces are plotted and all the info stored in the _contact_data_simulation.csv_ file in the current working directory. The two output figures in case of default simulation parameters will look something similar to the ones shown below.

Built-in Method             |  Explicit Method
:-------------------------:|:-------------------------:
![picture 5](../../images/7e24e7103502946d76db99ae2c4ad5121f32d4ee9559a286477ecbe18b3e8316.png)  |  ![picture 4](../../images/393f8a1ae3eac1c13f817267d3835cb9335ff5eba8214583a00f2e78bb8175ad.png)  
