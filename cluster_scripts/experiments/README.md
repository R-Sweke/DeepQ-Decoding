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

### Observing training progress

There are multiple ways of checking if the agent is learning (we assume that training happens on a SLURM cluster).

- Run `sacct -X` to see all jobs (for quickly checking if job was correctly submitted or is still running).
- Run `less /0.001/output_files/out_hash_0.001_x.out` to see `stdout` of one config (for checking how many steps were performed, current average reward, ...).
- Run `python3 tools/training_visualizer.py cluster_scripts/experiments/your_training_run/0.001` to visualize `training_history.json` of all configs for one error rate.
- Run `tensorboard --logdir=experiments/cluster_scripts/experiments/your_training_run/0.001/tensorboard_logs` to visualize training progress in `TensorBoard` [1]. Note you might need to do perform port binding in your `ssh` (Use: `ssh -L 16006:127.0.0.1:6006 cluster_name` to connect to TensorBoard on your local machine on port `16006`)

### Summary of changes (not complete)
- The `Environment.py` and the `Functions_Library.py` are not copied to every config but added to environment via `pip`. Make sure to install `lib` contents in your environment.
- It is possible to use MWPM now. Set `static_decoder` in `fixed_config` to `None`.

### References

[1] https://www.tensorflow.org/tensorboard/get_started