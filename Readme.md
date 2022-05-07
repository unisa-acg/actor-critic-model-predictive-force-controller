# Environment Identification

This repository collects the work conducted by a collaboration between UniSa, PoliMi and IDSIA. The repository is thought for students with the aim to provide an environment for the development of master theses projects and some research activities.

## Workflow

The code in this repository is organized in branches, while the workflow is inspired by [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html) (an alternative guide can be found [here](https://www.atlassian.com/it/git/tutorials/comparing-workflows/gitflow-workflow)). There are two **official branches**:

* `master`: this branch only contains stable releases, code that is distributed in various forms, like publication on GitHub, the Automatic Control Group's website or other channels;
* `develop`: this branch contains stable documented peer-reviewed development code, that passed tests (unit tests, integration tests, or other forms of tests); it is the branch from which to start to develop new features or fix bugs.

Only administrators (usually PhD students, researchers or professors of the group) can write the `develop` and `master` branches.

All the students (including those developing their master projects) contributing to the repository, as well as other external contributors, develop on separate **unofficial branches**, which can be:

* `bugfix/<name>`: branches used to correct bugs or integrate incomplete information;
* `feature/<name>`: branches used to develop new functions or documentation.

The `<name>` must be agreed with one of the administrators.

After an unofficial branch has been **separately tested and documented**, the contributor(s) can request the merge of the branch onto the `develop` through opening a **Pull Request (PR)**. PRs officially start the revision process. Reviewers can be administrators, but other people can serve as reviewers.

Reviewers may generate comments requiring modifications of the source code, configuration or documentation on the unofficial branches. Upon implementing all the comments, which is a responsibility of the contributor, the unofficial branch is merged onto the `develop` and officially enters the code base of the lab.

## Useful links and material

### Scrum

The development of this code base follows an incremental approach and, for this reason, agile methodologies well fit for it. Master thesis projects are developed using [Scrum](https://www.scrumguides.org/scrum-guide.html), but in-course projects might also benefit from using this paradigm.

### Git

This is a version-controlled git-based repo. If you don't know what version control and git are, or just need to refresh them, you can look at [this handbook](https://guides.github.com/introduction/git-handbook/).

### Object-oriented programming and design patterns

Most of the code in this repo is object-oriented and written in C++ and Python. Quality software often benefits from design patterns, that are constructs useful to solve common software design problems. This repo sometimes implements some of them, like singletons and factories, to mention some. You can give a look at some examples [here](https://refactoring.guru/design-patterns/cpp).

If you are not confident with C++ or Python, a lot of documentation is available online. If you need more detailed documentation about the language, please ask your teacher/supervisor.

### ROS tutorials

This code base is heavily based on the *Robot Operating System (ROS)* and its planning system *MoveIt!*. You can find a list of useful tutorials here below:

* [ROS](http://wiki.ros.org/ROS/Tutorials)
* [ROS2](https://index.ros.org/doc/ros2/Tutorials/)
* [MoveIt!](https://ros-planning.github.io/moveit_tutorials/)
* [ROS Control](http://wiki.ros.org/ros_control)
* [ROS Control presentation video](https://vimeo.com/107507546)
* [pluginlib](http://wiki.ros.org/pluginlib/Tutorials)

## Contributing to the development

### How to commit

Commits should be **small** and related to a specific set of changes that are **semantically related to each other**. Although unofficial branches allow for any committing style, short commits are beneficial to keep the repo clean and tidy. If you need to go back to a previous commit or making a-posteriori analyses of your code, finer granularity helps.

In case you need to make a big code refactoring, always remember that you can proceed by committing small incremental work that is still semantically self-contained.

Also, **think before committing!** You should design your commit before typing your `git commit` command, or even before modifying the code. This helps you focusing on the function you are going to implement and better organize your work.

In case you did not think enough, and you made an unfortunate mistake, please read [this guide](https://sethrobertson.github.io/GitFixUm/fixup.html) before trying to solve the problem yourself and possibly stacking additional (unsolvable) mistakes.

### Commit messages

Configure git for using the .gitmessage as commit template issuing the following command:

```bash
git config commit.template .gitmessage
```

this command configures git to use this template only for this project, if you like to configure git to use it for all project you should add the global flag as follows:

```bash
git config --global commit.template ~/.gitmessage
```

When writing commit messages, please use the following conventions

* ADD adding new feature
* FIX a bug
* DOC documentation only
* REF refactoring that doesn't include any changes in features
* FMT formatting only (spacing...)
* MAK repository related changes (e.g., changes in the ignore list)
* TST related to test code only

Use bullet lists for commits including more than one change. **See the latest commit messages for an example before making your first commit!**

### Debug

In order to acquire very basic information about integrating VSCode and ROS, visit [this page](https://medium.com/@tahsincankose/a-decent-integration-of-vscode-to-ros-4c1d951c982a). It is preparatory to understand the content of this section. In particular, the reader should know what a debug configuration in VSCode is and what pre-launch tasks are.

The focus of this section is on debugging ROS nodes, including test nodes. Like other C++ executables, they are characterized by a main function, that is the entry point of the binary executable. Unlike common C++ executables, ROS nodes usually require a ROS master to be up and running, therefore they often need to be launched with `roslaunch`. In fact, this python command takes care of launching a master before the actual node executable is launched. In addition, `roslaunch` can execute further operations, as demanded in the relating launch file, such as loading specific configuration files or placing parameters on the parameter server.

VSCode allows to debug C++ executables through a customizable debugger (usually GDB) and provides two configurations that are commonly referred to as `launch` and `attach`. The former tells VSCode to spawn the process to debug (only compiled executables can be run, which is not the case of `rosrun` and `roslaunch`), the latter tells VSCode to attach to a running process, which, therefore, must exist before the debugger is executed, which is not, in general, the case of our ROS nodes.

The solution lies in the pre-launch tasks, that are processes that can be executed by VSCode before the debugger is executed. Through a pre-launch task, we may run our `rosrun` or `roslaunch` command, so that the master executes, then the debugger takes care of launching the actual executable (or library, in case of plugins, dynamically-loaded libraries). Since VSCode will wait for the pre-launch task to complete before calling the debugger, and the master must not terminate before the node is executed, it is necessary to configure the following option in the pre-launch task:

```json
"isBackground": true
```

This will allow VSCode to run the pre-launch task in the background and will not wait for its termination before launching the debugger. Sometimes, the option above makes VSCode complain that the process cannot be tracked. If this happens, a `problemMatcher` needs to be configured:

```json
"problemMatcher": [
    {
        "pattern": [
            {
                "regexp": ".",
                "file": 1,
                "location": 2,
                "message": 3
            }
        ],
        "background":
        {
            "activeOnStart": true,
            "beginsPattern": ".",
            "endsPattern": ".",
        }
    }
]
```

The parameters used in the `problemMatcher` are completely random, but VSCode wants them to be defined anyway.

A complete example of a task configuration in `tasks.json` (to debug a test executable) is reported here below:

```json
"tasks": [
    {
        "label": "test_dynamic_programming_solver_totp",
        "type": "shell",
        "command": "roslaunch moveit_dp_redundancy_resolution test_dynamic_programming_solver_planar_2r.test debugger_attached:=true",
        "isBackground": true,
        "problemMatcher": [
            {
                "pattern": [
                    {
                        "regexp": ".",
                        "file": 1,
                        "location": 2,
                        "message": 3
                    }
                ],
                "background":
                {
                    "activeOnStart": true,
                    "beginsPattern": ".",
                    "endsPattern": ".",
                }
            }
        ]
    }
]
```

It is worth noticing, in the snippet above, the presence of a launch parameter called `debugger_attached`. This parameter tells `roslaunch` that the node should not be spawned, because it will by the debugger. This way, `roslaunch` will start the ROS master, load the needed parameters on the parameter server and will skip the actual execution of the node. In order for this to work, the launch file should have an `unless` keyword in the `node` or `test` tag, e.g., for node,

```xml
<node name="demo_totpr_node" pkg="moveit_dp_redundancy_resolution" type="demo_totpr_node" output="screen" unless="$(arg debugger_attached)"/>
```

The `configurations` in `launch.json` will look like this:

```json
"configurations": [
    {
        "name": "(gdb) Launch",
        "type": "cppdbg",
        "request": "launch",
        "program": "${workspaceFolder}/../devel/.private/moveit_dp_redundancy_resolution/lib/moveit_dp_redundancy_resolution/dynamic_programming_solver_time_optimal_planning-test",
        "args":[],
        "stopAtEntry": true,
        "cwd": "${workspaceFolder}",
        "environment": [],
        "internalConsoleOptions": "openOnSessionStart",
        "externalConsole": false,
        "preLaunchTask": "test_dynamic_programming_solver_totp",
        "MIMode": "gdb",
        "setupCommands": [
            {
                "description": "Enable pretty-printing for gdb",
                "text": "-enable-pretty-printing",
                "ignoreFailures": true
            },
        ]
    },
]
```

Running the configuration here called `(gdb) Launch` would suffice to attach the debugger to the process.

If you want to debug a plugin library, the process is the same, except that the `program` attribute of the configuration must point to the plugin library, e.g., for move group,

```json
"program": "/opt/ros/melodic/lib/moveit_ros_move_group/move_group"
```

## Authors

* **Vincenzo Petrone** - [UniSa - Automatic Control Group](http://www.automatica.unisa.it/)
* **Loris Roveda** - [IDSIA](https://www.idsia.ch/)

## Contributors

### Master students

* **Alessandro Pozzi** - [PoliMi](https://www.polimi.it/)
* **Luca Puricelli** - [PoliMi](https://www.polimi.it/)