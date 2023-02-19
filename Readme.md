# Experimental validation of an Actor-Critic Model Predictive Force Controller for robot-environment interaction tasks

This repository collects the work conducted by a collaboration between UniSa, PoliMi and IDSIA. The repository is thought as a support and demonstration material for the homonymous paper.

## Getting Started

This section explains how to setup the environment needed to launch the demos. First, MuJoCo needs to be installed, then some preliminary packages are required for the virtual environment creation.

### Install MuJoCo 2.1.0

In order to install MuJoCo you need to:

1. Create a hidden folder:

   ```sh
   cd
   mkdir .mujoco
   ```

2. Download MuJoCo library

   ```sh
   cd .mujoco
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

### Prerequisites

You will need Pip for the virtual environment packages installation, venv for the virtual environment and some other packages as prerequisites for Mujoco-Py. ```sudo``` can be needed for some installations.

* Pip

   ```
   apt install pip
   ```

* Packages needed for Mujoco-Py

   ```
   apt install libosmesa6-dev libgl1-mesa-glx libglfw3
   apt install patchelf gcc
   apt install python3-dev build-essential libssl-dev libffi-dev libxml2-dev
   apt install libxslt1-dev zlib1g-dev libglew-dev
   ```

* Python-venv

   ```
   apt install python3.8-venv
   ```

* [ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu)
  
### Clone the repository

Clone environment_identification repository

   ```
   cd
   git clone https://bitbucket.org/unisa-polimi-idsia/environment_identification/
   ```


### Setting up the virtual environment

Create a python [venv](https://linuxhint.com/python-virtualenv-tutorial/) and activate it

```
python3 -m venv venv --python=python3.8 # Create the venv
source venv/bin/activate # Activate the venv
```

Install via pip the required packages into the virtual environment

```
cd environment_identification
python3 -m pip install -r requirements.txt
```

### Substitute the base robosuite controllers

Substitute the files ```base_controller.py``` and ```osc.py``` in robosuite (substitute USERNAME with your user name in the command lines):

   ```sh
   cp dataset_generation_utilities/osc.py /home/USERNAME/venv/lib/python3.8/site-packages/robosuite/controllers # substitute USERNAME with your user name
   cp dataset_generation_utilities/base_controller.py /home/USERNAME/venv/lib/python3.8/site-packages/robosuite/controllers # substitute USERNAME with your user name
   ```


<!-- ### Clone robosuite

Robosuite is a simulation framework powered by the MuJoCo physics engine that offers a suite of benchmark environments.

1. Clone the robosuite repository:

   ```sh
   cd environment_identification
   git clone https://github.com/StanfordVL/robosuite.git
   ```

2. Rename robosuite folder to avoid path conflicts

   ```sh
   mv robosuite robosuite_master
   ```

3. Substitute the file osc.py in robosuite:

   ```sh
   cp dataset_generation/osc.py robosuite_master/robosuite/controllers
   ```

### Create and activate a Conda environment with the required packages


   ```sh
   cd environment_identification
   conda create --name env_identification
   conda activate env_identification
   pip install -r requirements.txt
   ``` -->

## Usage

Refer to the Readme.md in the [ACMPFC](/ACMPFC) folder.

<!-- ## List of Modules

| Package                                                                       | Functionality                                      |
| :------------------------------------------------------------------------- | :------------------------------------------------- |
| [MuJoCo Validation](/mujoco_validation) | Retrieve the contact info and forces happened during a simulation between each pair of bodies in contact and validate the built-in method that computes the contact forces. |
| [Trajectory Generation](/trajectory_generation) | Provide a pipeline to retrieve customizable 2D trajectories, keeping track of the various steps through .csv files. |
| [Dataset Generation](/dataset_generation) | Provide a module to retrieve the useful data from the simulations and process them in oder to generate a .csv dataset able to train a neural network. |
| [Main](/main) | Contains the main scripts used for this repository. | -->

## Authors

* **Vincenzo Petrone** - [UniSa - Automatic Control Group](http://www.automatica.unisa.it/)
* **Loris Roveda** - [IDSIA](https://www.idsia.ch/)

## Contributors

### Master students

* **Alessandro Pozzi** - [PoliMi](https://www.polimi.it/)
* **Luca Puricelli** - [PoliMi](https://www.polimi.it/)

[Back to top](#experimental-validation-of-an-actor-critic-model-predictive-force-controller-for-robot-environment-interaction-tasks)
