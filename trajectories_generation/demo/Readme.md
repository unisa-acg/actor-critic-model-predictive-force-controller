# demo

## Simulation description

This demo consists in trajectories and forces generation and resampling.
Each trajectory is composed by subtrajectories, which are line, circle and sine wave.
For each subtrajectory a constant or a ramp force is associated.
The outputs are both saved in .csv files and plotted.
The _traj_gen_ class requires some inputs, that are randomized by the _randomizer_.
The latter requires some limit parameters, that allow the creation of customized trajectories, listed here:

* the starting trajectories point;
* the operating zone defined by two points, the top right and the bottom left;
* the maximum number of subtrajectories;
* the maximum speed with which the trajectories can be covered;
* the maximum and minimum radius of the circles (if the dimensions go out of the operative area, the radius is automatically reduced);
* the maximum amplitude of the sine wave (if the dimensions go out of the operative area, the amplitude is automatically reduced);
* the maximum frequency of the sine wave;
* the maximum and minimum value of the reference force.

## How to launch it

Make sure you have followed the instructions in [getting started](../../Readme.md)

```sh
    cd 
    cd environment_identification
    source set_env.sh
    cd trajectories_generation/demos
    python3 demo_traj.py
```

## Expected output

The simulation will output two figures for each trajectory showing the comparison between the generated and resampled trajectory and force.
Moreover, the data are stored in the _output_ file in the _trajectories_generation_ directory.
