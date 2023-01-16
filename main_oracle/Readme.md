# main_oracle

This folder contains the main scripts of this repository.
In order to launch a test simulation showing the performance of ORACLE, together with the regularizers, refere to the Readme.md in the [script_for_test_oracle](script_for_test_oracle/) folder. In the latter, a script makes usage of a pre-trained neural network ensemble and a pre-generated dataset. If one would like to generate its own dataset and train the ensemble, it should refer to the respective folders [script_for_dataset_generation](script_for_dataset_generation/) and [script_for_nn_training](script_for_nn_training/).

## Folders list

| Folder                                                                       | Contents                                      |
| :------------------------------------------------------------------------- | :------------------------------------------------- |
| [Script for dataset generation](../main_oracle/script_for_dataset_generation) | Contain the script for the dataset creation. |
| [Script for nn training](../main_oracle/script_for_nn_training) | Contain the script for the training of the neural networks ensemble force estimator and the folder _example_data_, useful to test this script with a dataset already generated.|
| [Script for test oracle](../main_oracle/script_for_test_oracle) | Contain the script to test the ORACLE strategy and the folder _example_data_, useful to replicate the results shown in the attached paper. |