# Introduction 
<div id="top"></div>

To be added...

<!-- TABLE OF CONTENTS -->
<details>
  <summary> <b> Table of Contents </b> </summary>
  <ol>
    <li>
      <a href="#introduction">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#install-mujoco-210">Install MuJoCo 2.1.0</a></li>
        <li><a href="#install-mujoco-py">Install MuJoCo-Py</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#list-of-packages">List of Packages</a></li>
    <li><a href="#component-repos">Component Repos</a></li>
  </ol>
</details>
<br/>

# Getting Started

## Prerequisites

* [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
* Git
   ```sh
   sudo apt install git
   ```

## Install MuJoCo 2.1.0

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

To be added...

| Tutorial                                                                                                                                                  | Description                                                                                  |
| --------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| [ROS–Unity Integration](tutorials/ros_unity_integration/README.md)                                                                                        | A set of component-level tutorials showing how to set up communication between ROS and Unity |
| [URDF Importer](tutorials/urdf_importer/urdf_tutorial.md)                                                                                                 | Steps on using the Unity package for loading [URDF](http://wiki.ros.org/urdf) files          |
| [**New!**] [Visualizations](https://github.com/Unity-Technologies/ROS-TCP-Connector/blob/main/com.unity.robotics.visualizations/Documentation~/README.md) | Usage instructions for adding visualizations for incoming and outgoing ROS messages          |

# List of Packages

| Package                                                                                                                                                  | Functionality description                                                                                  |
| --------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| [MuJoCo Validation](mujoco_validation) | Set of utilities to retrieve the contact info and forces happened during a simulation between each pair of bodies in contact |
| [MuJoCo Panda](environment_identification/mujoco_panda) |  Franka Emika Panda implementation for Mujoco. Provides also force, motion and hybrid force/motion controllers for the robot   |


# Component Repos

| Repo                                                                       | Functionality                                      |
| :------------------------------------------------------------------------- | :------------------------------------------------- |
| [Mujoco Panda](https://github.com/justagist/mujoco_panda) | Franka Emika Panda implementation for Mujoco. Provides also force, motion and hybrid force/motion controllers for the robot |


<p align="right">(<a href="#top">back to top</a>)</p>