### Experiments

This directory contains everything to generate experiments to be run on a HPC cluster.
The experiments are driven by a `training_config.json` file that describes the hyper-parameters and other configs for a training run. 
These JSON files should be stored in `experiments/config`. 
To generate an experiment from the JSON file use the script in `/src/make_experiment.py`. Call the script from the root directory as follows
```
python3 src/make_experiment.py cluster_scripts/experiments/configs/example_training_config.json
```
The script will take care of copying all necessary scripts to the new directory and generate configurations for the parameter grids you defined in `example_training_config.json`.
To start the training just run `bash Start_Simulations.sh` and start the `python3 Controller.py` in a screen instance. 


### Summary of changes (not complete)
- The `Environment.py` and the `Functions_Library.py` are not copied to every config but added to environment via `pip`. Make sure to install `lib` contents in your environment.
- It is possible to use MWPM now. Set `static_decoder` in `fixed_config` to `None`.