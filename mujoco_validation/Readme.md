# mujoco_validation

Retrieve the contact info and forces happened during a simulation between each pair of bodies in contact and validate the built-in method that computes the contact forces.
This method is validated in two different situations, as shown in the demos. These are based on the functions contained in the src folder.


## Folders list

| Folder                                                                       | Contents                                      |
| :------------------------------------------------------------------------- | :------------------------------------------------- |
| [demos](/mujoco_validation/demos) | Contain the examples of different environments in which the cumulative contact forces between pair of bodies are validated and the information (penetration, velocity of deformation of the contact etc.) are extracted from the simulation. |
| [media](/mujoco_validation/media) | Contain the images produced by the demos.|
| [output](/mujoco_validation/output) | Contain the .csv files, in which, step by step, the contact information are stored. |
| [src](/mujoco_validation/src) | Contain the utilities functions that allow the demo to work. |


