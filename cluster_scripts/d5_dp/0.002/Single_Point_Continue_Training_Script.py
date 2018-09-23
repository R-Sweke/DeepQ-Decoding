# ------------ This script runs a training cycle for a single configuration point ---------------

import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator

import rl as rl
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger

import json

from Function_Library import *
from Environments import *

import copy
import gym
import sys
import os
import shutil
import datetime

# ---------------------------------------------------------------------------------------------

variable_config_number = sys.argv[1]
base_directory = sys.argv[2]

variable_configs_folder = os.path.join(base_directory, "./config_"+str(variable_config_number) + "/")
variable_configs_path = os.path.join(variable_configs_folder, "variable_config_"+variable_config_number + ".p" )
fixed_configs_path = os.path.join(base_directory, "../fixed_config.p")

fixed_configs = pickle.load( open(fixed_configs_path, "rb" ) )
variable_configs = pickle.load( open(variable_configs_path, "rb" ) )

all_configs = {}

for key in fixed_configs.keys():
    all_configs[key] = fixed_configs[key]

for key in variable_configs.keys():
    all_configs[key] = variable_configs[key]

if fixed_configs["static_decoder"]:
  static_decoder = load_model(os.path.join(base_directory, "../static_decoder"))
else:
  static_decoder = None


# ---------------------------------------------------------------------------------------------

def build_convolutional_nn(cc_layers,ff_layers, input_shape, num_actions):
    
    # cc_layers =[num_filters, kernel_size,strides]
    
    model = Sequential()
    model.add(Conv2D(filters=cc_layers[0][0], 
                     kernel_size=cc_layers[0][1], 
                     strides=cc_layers[0][2], 
                     input_shape=input_shape,
                     data_format='channels_first'))
    model.add(Activation('relu'))
    
    for j in range(1,len(cc_layers)):
            model.add(Conv2D(filters=cc_layers[j][0], 
                     kernel_size=cc_layers[j][1], 
                     strides=cc_layers[j][2], 
                     data_format='channels_first'))
            model.add(Activation('relu'))
            
    model.add(Flatten())
    
    for j in range(len(ff_layers)):
        model.add(Dense(ff_layers[j][0]))
        model.add(Activation('relu'))
        model.add(Dropout(rate=ff_layers[j][1]))

    model.add(Dense(num_actions))
    model.add(Activation('linear'))
         
    return model

# ---------------------------------------------------------------------------------------------

logging_path = os.path.join(variable_configs_folder,"training_history.json")
logging_callback = FileLogger(filepath = logging_path,interval = all_configs["print_freq"])

# --------------------------------------------------------------------------------------------

env = Surface_Code_Environment_Multi_Decoding_Cycles(d=all_configs["d"], 
    p_phys=all_configs["p_phys"], 
    p_meas=all_configs["p_meas"],  
    error_model=all_configs["error_model"], 
    use_Y=all_configs["use_Y"], 
    volume_depth=all_configs["volume_depth"],
    static_decoder=static_decoder)

# -------------------------------------------------------------------------------------------

memory_file = os.path.join(variable_configs_folder, "memory.p")
memory = pickle.load(open(memory_file,"rb"))

model = build_convolutional_nn(all_configs["c_layers"],all_configs["ff_layers"], env.observation_space.shape, env.num_actions)
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
               test_policy = test_policy,
               gamma = all_configs["gamma"],
               enable_dueling_network=all_configs["dueling"])  


dqn.compile(Adam(lr=all_configs["learning_rate"]))

weights_file = os.path.join(variable_configs_folder, "dqn_weights.h5f")
dqn.model.load_weights(weights_file)

# -------------------------------------------------------------------------------------------

now = datetime.datetime.now()
started_file = os.path.join(variable_configs_folder,"started_at.p")
pickle.dump(now, open(started_file, "wb" ) )

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

# --------------------------------------------------------------------------------------------

pickle.dump(dqn.memory, open(memory_file, "wb" ) )

# -------------------------------------------------------------------------------------------

model = build_convolutional_nn(all_configs["c_layers"],all_configs["ff_layers"], env.observation_space.shape, env.num_actions)
memory = SequentialMemory(limit=all_configs["buffer_size"], window_length=1)
policy = GreedyQPolicy(masked_greedy=True)
test_policy = GreedyQPolicy(masked_greedy=True)

# ------------------------------------------------------------------------------------------

dqn = DQNAgent(model=model, 
               nb_actions=env.num_actions, 
               memory=memory, 
               nb_steps_warmup=all_configs["learning_starts"], 
               target_model_update=all_configs["target_network_update_freq"], 
               policy=policy,
               test_policy=test_policy,
               gamma = all_configs["gamma"],
               enable_dueling_network=all_configs["dueling"])  


dqn.compile(Adam(lr=all_configs["learning_rate"]))
dqn.model.load_weights(weights_file)

# -------------------------------------------------------------------------------------------

nb_test_episodes = all_configs["testing_length"]
testing_history = dqn.test(env,nb_episodes = nb_test_episodes, visualize=False, verbose=2, interval=10, single_cycle=False)
results = testing_history.history["episode_lifetimes_rolling_avg"]

results_file = os.path.join(variable_configs_folder,"results.p")
pickle.dump(results, open(results_file, "wb" ) )
