# Introduction 
<div id="top"></div>

To be added...

<b>Table of Contents </b>
1. <a href="#getting-started">Getting Started</a>
   * <a href="#prerequisites">Prerequisites</a>
   * <a href="#installmujoco">Install Mujoco</a>
   * <a href="#installmujocopy">Install Mujoco-Py</a>
2. <a href="#usage">Usage</a>
3. <a href="#listofpackages">List of Packages</a>
4. <a href="#componentrepos">Component Repos</a>

# Getting Started
This section explains how to setup the environment needed to launch the demos

<!-- ----------------------------------------------------------------------- -->

## Prerequisites 

* [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
* Git
   ```sh
   sudo apt install git
   ```

## Install MuJoCo 2.1.0
<div id="installmujoco"></div>

1. Download MuJoCo library from [MuJoCo website](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz)
2. Create a hidden folder:
   ```sh
   mkdir /home/username/.mujoco
   ```

3. Extract the MuJoCo downloaded library into the hidden folder `.mujoco`
4. Add these lines to the `.bashrc` file:
   ```sh
   export LD_LIBRARY_PATH=/home/user_name/.mujoco/mujoco210/bin #substitute username with your username
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia #if nvidia graphic
   export PATH="$LD_LIBRARY_PATH:$PATH
   ```

5. Source the `.bashrc` file:
   ```sh
   source .bashrc
   ```

6. Test the installation:
   ```sh
   cd ~/.mujoco/mujoco210/bin
   ./simulate ../model/humanoid.xml
   ```

## Install MuJoCo-Py
<div id="installmujocopy"></div>

1. Create and activate a Conda environment for MuJoCo-Py:
   ```sh
   conda create --name mujoco_py python=3.8
   conda activate mujoco_py
   ```

2. Install the required packages:
   ```sh
   sudo apt update
   sudo apt install patchelf
   sudo apt install python3-dev build-essential libssl-dev libffi-dev libxml2-dev  
   sudo apt install libxslt1-dev zlib1g-dev libglew1.5 libglew-dev python3-pip
   ```

3. Clone the MuJoCo-Py repository and install the required pip packages:
   ```sh
   git clone https://github.com/openai/mujoco-py
   cd mujoco-py
   pip3 install -r requirements.txt
   pip3 install -r requirements.dev.txt
   pip3 install -e . --no-cache
   ```

4. Reboot your pc
5. Activate the Conda virtual environment and install MuJoCo-Py:
   ```sh
   conda activate mujoco_py
   sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
   pip3 install -U 'mujoco-py<2.2,>=2.1' 
   ```

6. Test the installation:
   ```sh
   cd examples
   python3 markers_demo.py
   ```

# Usage
<div id="usage"></div>

To be added...

| Tutorial                                                                                                                                                  | Description                                                                                  |
| --------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| [ROS–Unity Integration](tutorials/ros_unity_integration/README.md)                                                                                        | A set of component-level tutorials showing how to set up communication between ROS and Unity |


# List of Packages
<div id="listofpackages"></div>
| Package                                                                                                                                                  | Functionality description                                                                                  |
| --------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| [MuJoCo Validation](mujoco_validation) | Set of utilities to retrieve the contact info and forces happened during a simulation between each pair of bodies in contact |
| [MuJoCo Panda](environment_identification/mujoco_panda) |  Franka Emika Panda implementation for Mujoco. Provides also force, motion and hybrid force/motion controllers for the robot   |



# Component Repos
<div id="componentrepos"></div>

| Repo                                                                       | Functionality                                      |
| :------------------------------------------------------------------------- | :------------------------------------------------- |
| [Mujoco Panda](https://github.com/justagist/mujoco_panda) | Franka Emika Panda implementation for Mujoco. Provides also force, motion and hybrid force/motion controllers for the robot |


<p align="right">[<a href="#top">Back to top</a>]</p>
