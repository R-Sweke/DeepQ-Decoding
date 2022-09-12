import os
import sys
import json
import shutil

# Script to generate training folder structure from json training config file -------------------

if len(sys.argv != 2):
  print("Usage: python3 make_experiment.py <training_config.json>")
  sys.exit(1)
  
config_file = sys.argv[1]
with open(config_file, 'r') as f:
    config = json.load(f)

# 1. Create base directory in /cluster_scripts/experiments
rootdir = f"cluster_scripts/experiments/{config['config_dir']}"
os.mkdir(rootdir)

# 2. Copy config file to root
shutil.copyfile(config_file, f"{rootdir}/training_config.json")

# 3. Create folder structure and mandatory files
os.mkdir(f"{rootdir}/results")
open(f"{rootdir}/history.txt", "w").close()

with open(f"{rootdir}/current_error_rate.txt", "w") as f:
  f.write(f"{config['base_param_grid']['p_phys']}\n")

for phy in config["controller_params"]["p_phys_list"]:
  os.mkdir(f"{rootdir}/{phy}")
  os.mkdir(f"{rootdir}/{phy}/output_files")
