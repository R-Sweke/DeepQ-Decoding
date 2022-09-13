### Tools

This directory contains scripts to visualize training progress, copy training results to save best performing configurations at each error rate and create a template to run experiments on an HPC cluster.
**Note:** All scripts expect to be executed from the root directory (`../tools`).

- `copy_training_results` : Copies for each error rate important files from training directory to output directory (usually `trained_models`). See `trained_models` directory for example of what files are stored.
- `make_experiment` : Creates a directory structure based on JSON config usually stored in `cluster_scripts/experiments/configs`. It will automatically generate configs for the starting error rate based on the parameter grid. **Note**: screen session for the controller is not instantiated and training has to be started with `bash Start_Simulations.sh`.
- `training_visualizer`: For a given training directory and error rate, the training progress is plotted using the metrics collected for each config in `training_history.json`.