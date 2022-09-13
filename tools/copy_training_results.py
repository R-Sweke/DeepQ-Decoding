import sys
import os

if len(sys.argv) != 3:
    print("Usage: python3 copy_results.py <training_dir> <destination_dir>")
    sys.exit(1)

training_dir = sys.argv[1]  # folder where training data is stored
# path to folder where best configs should be stored (e.g. "trained_models")
destination_dir = sys.argv[2]

results_dir = os.path.join(training_dir, "results")

# Loop over results directory and extract best config at every error rate
best_config_at_error_rate = {}
for file in os.listdir(results_dir):
    if file.startswith("best_results_from"):
        error_rate = file.split("best_results_from_")[1].rsplit(".", 1)[0]
        with open(os.path.join(results_dir, file), "r") as f:
            lines = f.readline()
            best_config = lines.rstrip().split(":")[0]
            best_config_at_error_rate[error_rate] = best_config

# Create base folder
if not os.path.exists(destination_dir):
    os.mkdir(destination_dir)

file_list = ["all_results.p", "final_dqn_weights.h5f",
             "training_history.json"]
for error, config in best_config_at_error_rate.items():
    config_file_list = file_list + [f"variable_config_{config}.p"]
    config_src_path = os.path.join(training_dir, error, f"config_{config}")
    config_dest_path = os.path.join(destination_dir, error)
    # Create folder for error rate in destination
    if not os.path.exists(config_dest_path):
        os.makedirs(config_dest_path, exist_ok=True)
        # Copy files specified in file list
        for datafile in config_file_list:
            src = os.path.join(config_src_path, datafile)
            dest = os.path.join(config_dest_path, datafile)
            os.popen(f"cp {src} {dest}")

# Copy fixed config to destination
fixed_config_src = os.path.join(training_dir, "fixed_config.p")
fixed_config_dst = os.path.join(destination_dir, "fixed_config.p")
os.popen(f"cp {fixed_config_src} {fixed_config_dst}")
