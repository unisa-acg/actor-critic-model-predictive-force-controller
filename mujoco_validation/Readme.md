# Introduction 
<div id="top"></div>

To be added...

<b>Table of Contents</b>

* <a href="#getting-started">Getting Started</a>
   * <a href="#prerequisites">Prerequisites</a>
   * <a href="#installmujoco">Install Mujoco</a>
   * <a href="#installmujocopy">Install Mujoco-Py</a>
* <a href="#usage">Usage</a>
* <a href="#listofpackages">List of Packages</a>
* <a href="#componentrepos">Component Repos</a>

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

| Demo                                                                                                                                                  | Description                                                                                  |
| --------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| [Contact forces and info retrieval](demos/contact_info_retrieval/Readme.md)                                                                                        | Example of different environments in which the cumulative contact forces between pair of bodies and the info (penetration, velcoity of deformation of the contact etc.) are stored in .csv file for later analysis |


# List of Packages
<div id="listofpackages"></div>

| Package                                                                       | Functionality                                      |
| :------------------------------------------------------------------------- | :------------------------------------------------- |
| MuJoCo Validation | Set of utilities to retrieve the contact info and forces happened during a simulation between each pair of bodies in contact |
| MuJoCo Panda |  See <a href="#componentrepos">Component Repos</a> for more info   |

<!-- TODO: give link to each package folder -->

# Component Repos
<div id="componentrepos"></div>

| Repo                                                                       | Functionality                                      |
| :------------------------------------------------------------------------- | :------------------------------------------------- |
| [Mujoco Panda](https://github.com/justagist/mujoco_panda) | Franka Emika Panda implementation for Mujoco. Provides also force, motion and hybrid force/motion controllers for the robot |


<p align="right">[<a href="#top">Back to top</a>]</p>
