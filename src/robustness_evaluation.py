import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf

from deepq.Function_Library import *
from deepq.Environments import *

import pickle
import sys
import os
import random
import pprint

# ---------------------------------------------------------------------------------------------

RANDOM_SEED = 0
os.environ['PYTHONHASHSEED']=str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)


# -----------------------------------------------------------------------------------------------

if len(sys.argv) != 5:
  print("Usage: python3 src/robustness_evaluation.py <training_dir> <error_rate> <config> <outputdir>")
  sys.exit(1)

training_dir = sys.argv[1] # path to trained model configs
error_rate = sys.argv[2] # error rate at which model was trained
config = sys.argv[3] # incumbent config number
output_dir = sys.argv[4] # output directory to store pickle files of test results

# -----------------------------------------------------------------------------------------------

def load_agent_config(trained_model_dir: str, config: str, error_rate: str):
    fixed_configs_path = os.path.join(trained_model_dir, "fixed_config.p")
    variable_configs_path = os.path.join(
        trained_model_dir, error_rate, f"variable_config_{config}.p")
    model_weights_path = os.path.join(
        trained_model_dir, error_rate, "final_dqn_weights.h5f")

    fixed_configs = pickle.load(open(fixed_configs_path, "rb"))
    variable_configs = pickle.load(open(variable_configs_path, "rb"))

    all_configs = {}
    for key in fixed_configs.keys():
        all_configs[key] = fixed_configs[key]
    for key in variable_configs.keys():
        all_configs[key] = variable_configs[key]
    all_configs["model_weights_path"] = model_weights_path
    return all_configs

all_configs = load_agent_config(training_dir, config, error_rate)
pprint.pprint(all_configs)

# --- Build env ---------------------------------------------------------------------------------

# For Heatmap noise we load heatmap from pickled file
scaling_factor = 1.6
with open("heatmap_experiments/a1000_t50_s5_dp_c2/heatmaps.p", "rb") as fp:
  heatmap_data = pickle.load(fp)
  # normalize heatmap
  p = heatmap_data["heatmap"]/np.sum(heatmap_data["heatmap"])
  # find factor so that max value in gaussian corresponds to depolarizing p_phys
  eps_p = scaling_factor*all_configs['p_phys']/np.max(p)
  # prepare kwargs dictionary for NoiseFactory
  kwargs = {
    "heat" : p*eps_p, 
  }

# noise_model = NoiseFactory("HEAT", all_configs["d"], all_configs["p_phys"], **kwargs).generate()
# noise_model = NoiseFactory(all_configs["error_model"], all_configs["d"], all_configs["p_phys"]).generate()
noise_model = XNoise(all_configs["d"], all_configs["p_phys"]/3, context='DP')

env = Surface_Code_Environment_Multi_Decoding_Cycles(
    d=all_configs["d"],
    p_meas=all_configs["p_meas"],
    noise_model=noise_model,
    use_Y=all_configs["use_Y"],
    volume_depth=all_configs["volume_depth"],
    static_decoder=None,
)

# --- Build model --------------------------------------------------------------------------------

dqn = build_eval_agent_model(all_configs, env)

# --- Model evaluation ---------------------------------------------------------------------------

trained_at = all_configs["p_phys"]
num_to_test = 20
error_rates = [j*0.001 for j in range(1, num_to_test + 1)]
thresholds = [1/p for p in error_rates]
nb_test_episodes = all_configs["testing_length"]
all_results = {}


keep_evaluating = True
count = 0
while keep_evaluating:
    # err_rate = error_rates[count]
    err_rate = 0.007
    env.noise_model.p_phys = err_rate
    # env.noise_model.set_physical_error_rate(err_rate)
    env.p_meas = err_rate

    dict_key = str(err_rate)[:5]

    testing_history = dqn.test(env, nb_episodes=nb_test_episodes,
                             visualize=False, verbose=2, interval=10, single_cycle=False)
    results = testing_history.history["episode_lifetimes_rolling_avg"]
    final_result = results[-1:][0]
    all_results[dict_key] = final_result

    if abs(trained_at - err_rate) < 1e-6:
        results_file = os.path.join(output_dir, "results.p")
        pickle.dump(results, open(results_file, "wb"))

    to_beat = thresholds[count]
    if final_result < to_beat or count == (num_to_test - 1):
        keep_evaluating = False

    count += 1

all_results_file = os.path.join(output_dir, "all_results.p")
pickle.dump(all_results, open(all_results_file, "wb"))