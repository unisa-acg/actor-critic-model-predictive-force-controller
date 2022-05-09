#! /bin/bash

export MJ_PANDA_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$MJ_PANDA_PATH:$PYTHONPATH

export MUJOCO_VALIDATION_PATH="$( cd "$( dirname "mujoco_validation" )" && pwd )"
export PYTHONPATH=$MUJOCO_VALIDATION_PATH:$PYTHONPATH

echo -e "Setting MJ_PANDA_PATH=$MJ_PANDA_PATH\n"
