# Environment Identification

This repository collects the work conducted by a collaboration between UniSa, PoliMi and IDSIA. The repository is thought for students with the aim to provide an environment for the development of master theses projects and some research activities.

## Getting Started

This section explains how to setup the environment needed to launch the demos.

### Prerequisites

* [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
* Git

   ```sh
   sudo apt install git
   ```

* Pip

   ```sh
   sudo apt install pip
   ```

* Packages needed for Mujoco-Py

   ```sh
   sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
   sudo apt install patchelf gcc
   sudo apt install python3-dev build-essential libssl-dev libffi-dev libxml2-dev
   sudo apt install libxslt1-dev zlib1g-dev libglew-dev
   ```

### Install MuJoCo 2.1.0

MuJoCo is a physics engine that aims to facilitate research where fast and accurate simulation of articulated structures interacting with their environment is needed.

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

### Clone environment_identification

1. Clone environment_identification repository and install the required pip packages:

   ```sh
   cd
   git clone https://bitbucket.org/unisa-polimi-idsia/environment_identification/
   ```

2. Create and activate a Conda environment with the required packages:

   ```sh
   cd environment_identification
   conda create --name env_identification
   conda activate env_identification
   pip install -r requirements.txt
   ```

## List of Modules

| Package                                                                       | Functionality                                      |
| :------------------------------------------------------------------------- | :------------------------------------------------- |
| [MuJoCo Validation](/mujoco_validation) | Retrieve the contact info and forces happened during a simulation between each pair of bodies in contact and validate the built-in method that computes the contact forces. |
| [Trajectory Generation](/trajectory_generation) | Provide a pipeline to retrieve customizable 2D trajectories, keeping track of the various steps through .csv files. |

## Authors

* **Vincenzo Petrone** - [UniSa - Automatic Control Group](http://www.automatica.unisa.it/)
* **Loris Roveda** - [IDSIA](https://www.idsia.ch/)

## Contributors

### Master students

* **Alessandro Pozzi** - [PoliMi](https://www.polimi.it/)
* **Luca Puricelli** - [PoliMi](https://www.polimi.it/)

[Back to top](#environment-identification)
