import sys
import os

if len(sys.argv) != 3:
        print("Usage: python3 copy_results.py error_rate destination")
        sys.exit(1)

error_rate = sys.argv[1]
destination_path = sys.argv[2]

results_path = f"results/best_results_from_{error_rate}.txt"

best_config = ""

with open(results_path, "r") as f:
        lines = f.readline()
        best_config = lines.rstrip().split(":")[0]


filelist = ["all_results.p", "final_dqn_weights.h5f", "training_history.json", f"variable_config_{best_config}.p"]
results_path = f"{error_rate}/config_{best_config}/"
for datafile in filelist:
        res_path = results_path + datafile
        os.popen(f"cp {res_path} {destination_path}")