# Getting Started

This section explains how to setup the environment needed to launch the demos.

<!-- ----------------------------------------------------------------------- -->

## Prerequisites

* [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
* Git

   ```sh
   sudo apt install git
   ```

## Install MuJoCo 2.1.0

MuJoCo is a physics engine that aims to facilitate research where fast and accurate simulation is needed.

1. Create a hidden folder:

   ```sh
    cd
    mkdir .mujoco
   ```

2. Download MuJoCo library from [MuJoCo website](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz)

   ```sh
    wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
    ```

3. Extract the MuJoCo downloaded library into the hidden folder `.mujoco`

    ```sh
    tar -xf mujoco210-linux-x86_64.tar.gz -C .
    rm mujoco210-linux-x86_64.tar.gz
    ```

4. Add these lines to the `.bashrc` file:

   ```sh
   export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin #substitute username with your username
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia #if nvidia graphic
   export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
   export PATH="$LD_LIBRARY_PATH:$PATH"
   ```

5. Source the `.bashrc` file:

   ```sh
   cd
   source .bashrc
   ```

6. Test the installation:

   ```sh
   cd ~/.mujoco/mujoco210/bin
   ./simulate ../model/humanoid.xml
   ```

## Install Robosuite

1. Activate the Conda virtual environment

   ```sh
   conda activate mujoco_py
   ```

2. Clone the Robosuite repository and install the required pip packages:

   ```sh
   git clone https://github.com/StanfordVL/robosuite.git
   cd robosuite
   pip3 install -r requirements.txt
   ```

3. Test the installation:

   ```sh
   python robosuite/demos/demo_random_action.py
   ```

## List of Modules

| Package                                                                       | Functionality                                      |
| :------------------------------------------------------------------------- | :------------------------------------------------- |
| [MuJoCo Validation](https://github.com/justagist/mujoco_validation) | Retrieve the contact info and forces happened during a simulation between each pair of bodies in contact and validate the built-in method that computes the contact forces. |
| [Mujoco Panda](https://github.com/justagist/mujoco_panda) | Franka Emika Panda implementation for Mujoco. Provides also force, motion and hybrid force/motion controllers for the robot. |
| [Workspace Trajectory Commander](https://github.com/justagist/workspace_trajectory_commander) | Provide a pipeline to retrieve customizable 2D trajectories, keeping track of the various steps through .csv files. |

<!-- TODO: give link to each package folder -->

<p align="right">[<a href="#top">Back to top</a>]</p>
