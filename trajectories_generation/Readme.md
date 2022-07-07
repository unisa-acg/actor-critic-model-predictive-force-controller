# Demo - Dataset creation

## Simulation description

This simulation consists in an environment composed of a Franka Emika Panda robot, with attached at the end-effector a sphere, and a fixed plane. The robot follows _N_ random trajectories and it is controlled via a position controller. The eventual contact forces between the sphere and the table are plotted at the end of the simulation.

## How to launch it

Clone the repository (if not done previously) and inside the main folder execute the following commands to source the environment and launch the demo:

```sh
    source set_env.sh
    cd demos/dataset_retrieval
    python3 demo_dataset_creation.py
```

## Expected output

The simulation will output some figures showing the contact forces history during the simulation, as the one reported below as example. 

<!-- ![picture 1](../../images/Figure_1.png)-->

![pic1](../../images/Figure_1.png)
