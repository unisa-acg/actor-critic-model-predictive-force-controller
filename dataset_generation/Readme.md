# dataset_generation

Retrieve the useful contact information happened during the simulations between the robot end-effector and the table and process the data obtained, generating a dataset.
During the simulation, some tools, contained in the folder _utilities_, are exploited to check its correct performance, plotting some information.
Finally the processed and unprocessed data are plotted to compare them.

## Folders list

| Folder                                                                       | Contents                                      |
| :------------------------------------------------------------------------- | :------------------------------------------------- |
| [main](/dataset_generation/main) | Contain the file used to generate the .csv dataset. In addition, plot simulation information and store the data of the most important steps in .csv files |
| [output](/dataset_generation/output) | Contain the .csv files, divided into four subfolders. |
| [src](/dataset_generation/src) | Contain the utilities functions that allow the demo to work. |
