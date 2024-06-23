# Description

Source code for experiments described in Erin J. Talvitie, Zilei Shao, Huiying Li, Jinghan Hu, Jacob Boerma, Rory Zhao, Xintong Wang. Bounding-Box Inference for Error-Aware Model-Based Reinforcement Learning. In _Proceedings of the 1st Reinforcement Learning Conference (RLC)_, 2024. Please see the paper for descriptions of algorithms and experiments.

# Disclaimer/Apology

This is classic "research code." It was developed in a disorganized and ad-hoc manner as the needs of the project dictated, not via a disciplined design process. It was not developed to be general purpose, but rather to support specific experiments. As such, though I have tried to clean it up a little bit, it is very likely that it contains poor/puzzling design choices, vestigial appendages, and other oddities and/or flaws. It is also poorly documented. Please forgive my mess. The main purpose of this release is to permit reproduction of the empirical results and to archive the source code. If you would like to adapt some of this code for your own purposes, feel free to contact me with questions and I will do my best to help!

# Compiling

This project uses CMake. To compile, navigate to the top-level directory of the repository. Then execute the following shell commands:

```
mkdir build; cd build
cmake ../
cmake --build .
```

This will create the executable _planning_ in the _bin_ directory.

# Generating Results

If you wish to run your own custom experiments, from the _bin_ directory you can run
```
./planning -h
```
to see the command-line arguments.

If you wish to run the experiments described in the paper, there are multiple Python scripts in the _bin_ directory to help. To reproduce the results in the paper, follow these steps:

## Parameter Sweep
Note that, if you wish to simply run final experiments with the metaparameter values used in the paper, you can skip this step.

First, generate configuration files for the parameter sweeps:
```
python3 generate_gor_cgfs.py --path <PATH FOR CONFIGS> --num_trials 10 --gen_manifest
python3 generate_acro_cfgs.py --path <PATH FOR CONFIGS> --num_trials 10 -- gen_manifest
```

Next, run the parameter sweep. Each job should be run from the _bin_ directory as:
```
./planning --config <CONFIG FILE>
```
For what it's worth, we used GNU Parallel [1] to organize our jobs:
```
cat <PATH WITH CONFIGS>/manifest.txt | parallel --jobs <NUM SIMULTANEOUS JOBS> --eta --progress ./planning --config
```

Finally, select the best performing metaparameter settings:
```
python3 best_gor_params.py --path <PATH WITH RESULTS> > gor.params
python3 best_acro_params.py --path <PATH WITH RESULTS> > acro.params
```

##  Final Experiments
Note that the _bin_ directory in the repository contains the _gor.params_ and _acro.params_ files that were generated by our experiments. You can use those to run experiments with our selected mataparameter values or perform your own parameter sweep using the instructions in the previous section.

First, generate configuration files using the selected configuration files:
```
python3 generate_selected_configs.py gor.params --output_path <PATH FOR FINAL CONFIGS> --num_trials 50 --first_seed 11 --gen_manifest
python3 generate_selected_configs.py acro.params --output_path <PATH FOR FINAL CONFIGS> --num_trials 50 --first_seed 11 --gen_manifest
```

Then, run the experiemnts. Each job should be run from the _bin_ directory as:
```
./planning --config <CONFIG FILE>
```
For what it's worth, we used GNU Parallel [1] to organize our jobs:
```
cat <PATH WITH FINAL CONFIGS>/manifest.txt | parallel --jobs <NUM SIMULTANEOUS JOBS> --eta --progress ./planning --config
```

# Brief Guide to the Source Code

If you want to know about...
- the RL algorithm: check out _src/rl/QLearner.*pp_
- the Q-function features: check out _src/rl/TileCodingQFunction.*pp_
- the environments: check out _src/rl/environmentsAcrobot.*pp_ and _src/rl/environments/GoRight.*pp_ 
- the hand-coded models: check out _src/rl/environments/GoRight.*pp_ (the ```GoRightUncertain``` class)
- the regression tree models: check out _src/rl/models/IncDTModel.*pp_ and _src/rl/models/FastIncModelTree.*pp_
- the neural network models: check out _src/rl/models/NNModel.*pp_
- the main loop: check out _src/planning.cpp

# References

[1] O. Tange. Gnu parallel - the command-line power tool. _;login: The USENIX Magazine_, 36(1):42–47, Feb 2011. URL http://www.gnu.org/s/parallel.

