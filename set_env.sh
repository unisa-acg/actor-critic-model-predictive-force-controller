#! /bin/bash

export MUJOCO_VALIDATION_PATH="$( cd "$( dirname "mujoco_validation" )" && pwd)"
export PYTHONPATH=$MUJOCO_VALIDATION_PATH:$PYTHONPATH

export MJ_PANDA_PATH="$( cd "$( dirname "mujoco_panda_master/mujoco_panda" )" && pwd)"
export PYTHONPATH=$MJ_PANDA_PATH:$PYTHONPATH

export ROBOSUITE_PATH="$( cd "$( dirname "robosuite_master/robosuite" )" && pwd)"
export PYTHONPATH=$ROBOSUITE_PATH:$PYTHONPATH

echo -e "Setting MUJOCO_VALIDATION_PATH=$MUJOCO_VALIDATION_PATH\n"
echo -e "Setting MUJOCO_PANDA_PATH=$MJ_PANDA_PATH\n"
echo -e "Setting ROBOSUITE_PATH=$ROBOSUITE_PATH\n"
echo -e "All set!\n"
