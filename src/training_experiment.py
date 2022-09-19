# ------------ This script runs a training cycle for a single configuration point ---------------

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger

from deepq.Function_Library import *
from deepq.Environments import *

import os
import datetime
import pickle

# ---------------------------------------------------------------------------------------------

fixed_config = {"d": 7,
                "use_Y": False,
                "train_freq": 1,
                "batch_size": 32,
                "print_freq": 50,
                "rolling_average_length": 1000,
                "stopping_patience": 1000,
                "error_model": "DP",
                "c_layers": [[64, 3, 2], [32, 2, 1], [32, 2, 1]],
                "ff_layers": [[512, 0.2]],
                "max_timesteps": 1000000,
                "volume_depth": 7,
                "testing_length": 101,
                "buffer_size": 50000,
                "dueling": True,
                "masked_greedy": False,
                "static_decoder": False}  # per default MWPM is used.

variable_config = {'p_phys': 0.001, 'p_meas': 0.001, 'success_threshold': 100000, 'learning_starts': 1000, 'learning_rate': 0.0001,
                   'exploration_fraction': 100000, 'max_eps': 1.0, 'target_network_update_freq': 2500, 'gamma': 0.99, 'final_eps': 0.04}

# ---------------------------------------------------------------------------------------------

all_configs = {}
for key in fixed_config.keys():
    all_configs[key] = fixed_config[key]

for key in variable_config.keys():
    all_configs[key] = variable_config[key]

static_decoder = None

# ---------------------------------------------------------------------------------------------

logging_path = os.path.join(os.getcwd(), "training_history.json")
logging_callback = FileLogger(
    filepath=logging_path, interval=all_configs["print_freq"])

# --------------------------------------------------------------------------------------------

noise_model = NoiseFactory(all_configs["error_model"], all_configs["d"], all_configs["p_phys"]).generate()

env = Surface_Code_Environment_Multi_Decoding_Cycles(d=all_configs["d"],
                                                     p_meas=all_configs["p_meas"],
                                                     noise_model=noise_model,
                                                     use_Y=all_configs["use_Y"],
                                                     volume_depth=all_configs["volume_depth"],
                                                     static_decoder=static_decoder)

# -------------------------------------------------------------------------------------------

model = build_convolutional_nn(
    all_configs["c_layers"], all_configs["ff_layers"], env.observation_space.shape, env.num_actions)
memory = SequentialMemory(limit=all_configs["buffer_size"], window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(masked_greedy=all_configs["masked_greedy"]),
                              attr='eps', value_max=all_configs["max_eps"],
                              value_min=all_configs["final_eps"],
                              value_test=0.0,
                              nb_steps=all_configs["exploration_fraction"])
test_policy = GreedyQPolicy(masked_greedy=True)

# ------------------------------------------------------------------------------------------

dqn = DQNAgent(model=model,
               nb_actions=env.num_actions,
               memory=memory,
               nb_steps_warmup=all_configs["learning_starts"],
               target_model_update=all_configs["target_network_update_freq"],
               policy=policy,
               test_policy=test_policy,
               gamma=all_configs["gamma"],
               enable_dueling_network=all_configs["dueling"])


dqn.compile(Adam(lr=all_configs["learning_rate"]))

# -------------------------------------------------------------------------------------------

now = datetime.datetime.now()
started_file = os.path.join(os.getcwd(), "started_at.p")
pickle.dump(now, open(started_file, "wb"))

history = dqn.fit(env,
                  nb_steps=all_configs["max_timesteps"],
                  action_repetition=1,
                  callbacks=[logging_callback],
                  verbose=2,
                  visualize=False,
                  nb_max_start_steps=0,
                  start_step_policy=None,
                  log_interval=all_configs["print_freq"],
                  nb_max_episode_steps=None,
                  episode_averaging_length=all_configs["rolling_average_length"],
                  success_threshold=all_configs["success_threshold"],
                  stopping_patience=all_configs["stopping_patience"],
                  min_nb_steps=all_configs["exploration_fraction"],
                  single_cycle=False)
