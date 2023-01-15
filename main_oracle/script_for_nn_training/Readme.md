# force_estimator_training

This script allows to training the neural networks ensemble E-NNs, called force estimator.
It has to be used once a dataset, with its normalization file, is built, using the script in folder [script_for_dataset_generation](../script_for_dataset_generation). Otherwise an example dataset ([sim_data_proc_example](../script_for_nn_training/example_data/sim_data_proc_example.csv), [norm_data_example](../script_for_nn_training/example_data/norm_data_example.csv)) can be used to try this code.
It takes in input x_f, the output of the PI force controller, and xf_dot, the derivative of the output of the PI force controller, linking them with the output, the contact force f_z_out.

## How to launch it

Make sure you have followed the instructions in [getting started](../../Readme.md)

```sh
    cd 
    cd environment_identification
    source set_env.sh
    cd test_oracle/script_for_nn_training
    python3 force_estimator_training.py
```

## Expected output

The simulation will generate three main outputs: the _logs_ folder, the _ouput_ folder, and five figures.

* The _logs_ folder contains the convergence data of the training;
* The _ouput_ folder contains the trained E-NNs;
* The figures represent:
  1. the prediction of the E-NNs compared to the actual one, for the verification dataset;
  2. the prediction error versus x_f and xf_dot (3d plot);
  3. the prediction error versus x_f;
  4. the prediction error versus xf_dot;
  5. x_f versus xf_dot;
