import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import pandas as pd
import seaborn as sb

import sys
import os
import math
import pickle
import json

# Script to plot relevant metrics for all configs at certain error rate -------------------
#
# TODO: script should offer argument to change output_dir
# TODO: script should offer filter to remove metrics
# TODO: script should offer filter to specify configs


if len(sys.argv) != 3:
  raise ValueError("Usage: python3 training_visualizer.py <error_rate> <output_dir>")

rootdir = sys.argv[1] # error rate directory for which we visualize config training metrics
error_rate = rootdir.split('/')[-1]
outputdir = sys.argv[2]

# get fixed config used by all configs for this error rate
fixed_config_dir = rootdir.rsplit('/',1)[0]
fixed_config_fd = open(os.path.join(fixed_config_dir, "fixed_config.p"), "rb")
fixed_config = pickle.load(fixed_config_fd)

# create folder structure
dir_exist = False
try:
  decoder = "nn" if fixed_config["static_decoder"] else "mwpm"
  outputdir = f"{error_rate}_d{fixed_config['d']}_bs{fixed_config['batch_size']}_st{fixed_config['stopping_patience']}_{decoder}_metrics"
  os.mkdir(outputdir)
except FileExistsError:
  print("Warning: Output directory already exists, will overwrite files")
  overwrite = input("Continue (y/n)? ")
  if overwrite.lower() == "n":
    sys.exit(1)
  dir_exist = True
except FileNotFoundError:
  raise FileNotFoundError("Parent directory doesn't exist")
# create folder structure if directory didn't exist yet
if not dir_exist:
  os.mkdir(os.path.join(outputdir, "figs"))
  os.mkdir(os.path.join(outputdir, "variable_config"))

# write fixed config to directory
with open(os.path.join(outputdir, "fixed_config.json"), "w") as f:
  json.dump(fixed_config, f)

# loop over all configs and load json training history into dictionary
# and write variable config into folder
data = {}
for dir in os.listdir(rootdir):
  if dir.startswith('config_'):
    training_history = os.path.join(rootdir, dir, "training_history.json")
    data[dir] = pd.read_json(training_history)

    variable_config_fd = open(os.path.join(rootdir, dir, f"variable_{dir}.p"), "rb")
    variable_config = pickle.load(variable_config_fd)
    with open(os.path.join(outputdir, "variable_config", f"variable_{dir}.json"), "w") as f:
      json.dump(variable_config, f)

metrics = [
          ("episode_lifetimes_rolling_avg", "episode lifetime roll. avg."),
          ("episode_reward", "episode reward"),
          ("loss", "loss"),
          ("nb_episode_steps", "steps per episode"),
          ("duration", "episode duration in secods"),
          ("mean_q", "mean q-value"),
          ("mean_eps", "mean eps-value")
          ]

# computing layout of subplot matrix
num_configs = len(data.keys())
rows = math.floor(math.sqrt(num_configs))
cols = math.ceil(math.sqrt(num_configs))

sb.set_style("darkgrid")
loc = plticker.MaxNLocator(nbins=6)
for metric, ylabel in metrics:
  fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True)
  fig.set_figheight(15)
  fig.set_figwidth(15)
  for idx, config in enumerate(data.keys()):
    c_row = idx % rows
    c_col = idx // rows
    axs[c_row,c_col].plot(data[config][metric])
    axs[c_row,c_col].set_title(config)
    axs[c_row,c_col].grid(visible=True)
    axs[c_row,c_col].tick_params(axis='x', rotation=90)
    axs[c_row,c_col].xaxis.set_major_locator(loc)
  fig.suptitle(f"{error_rate} {metric}")
  fig.supxlabel("episodes")
  fig.supylabel(ylabel)
  fig.tight_layout()
  fig.savefig(f"{outputdir}/figs/{error_rate}_{metric}.png", dpi=400)