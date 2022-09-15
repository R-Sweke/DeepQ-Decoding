import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from deepq.Environments import *
from deepq.Function_Library import *

from tqdm import trange

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import (
    GreedyQPolicy,
)

import os
import sys
import pickle
import pprint


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


def build_agent_model(all_configs, env):
    model = build_convolutional_nn(
        all_configs["c_layers"],
        all_configs["ff_layers"],
        env.observation_space.shape,
        env.num_actions,
    )
    memory = SequentialMemory(
        limit=all_configs["buffer_size"], window_length=1)
    policy = GreedyQPolicy(masked_greedy=True)
    test_policy = GreedyQPolicy(masked_greedy=True)

    dqn = DQNAgent(
        model=model,
        nb_actions=env.num_actions,
        memory=memory,
        nb_steps_warmup=all_configs["learning_starts"],
        target_model_update=all_configs["target_network_update_freq"],
        policy=policy,
        test_policy=test_policy,
        gamma=all_configs["gamma"],
        enable_dueling_network=all_configs["dueling"],
    )

    dqn.compile(Adam(lr=all_configs["learning_rate"]))
    dqn.model.load_weights(all_configs["model_weights_path"])
    return dqn


def sample_syndrome_volume(env, hidden_state):
    syndromes = []
    errors = []
    cost = 0. # keeps track of error probability
    for _ in range(env.d):
        error = generate_error(env.d, env.p_phys, env.error_model)
        cost += np.count_nonzero(error)
        new_hidden_state = obtain_new_error_configuration(hidden_state, error)
        current_true_syndrome = generate_surface_code_syndrome_NoFT_efficient(
            new_hidden_state, env.qubits
        )
        current_faulty_syndrome = generate_faulty_syndrome(
            current_true_syndrome, env.p_meas)
        errors.append(error)
        syndromes.append(current_faulty_syndrome)
    return {"errors": errors, "syndromes": syndromes, "cost": cost}


# --- Load agent config and weights ---------------------------------------------------------------------
if len(sys.argv) != 5:
  print("Usage: python3 src/robustness_checks/local_search.py <trained_model_dir> <error_rate> <config> <max_transitions>")
  sys.exit(1)

trained_model_dir = sys.argv[1]  # path to trained model configs
error_rate = sys.argv[2]  # error rate at which model was trained
config = sys.argv[3]  # incumbent config number
max_transitions = int(sys.argv[4])  # max transitions in "MDP"

all_configs = load_agent_config(trained_model_dir, config, error_rate)
pprint.pprint(all_configs)

# --- Build env ---------------------------------------------------------------------------------

env = Surface_Code_Environment_Multi_Decoding_Cycles(
    d=all_configs["d"],
    p_phys=all_configs["p_phys"],
    p_meas=all_configs["p_meas"],
    error_model=all_configs["error_model"],
    use_Y=all_configs["use_Y"],
    volume_depth=all_configs["volume_depth"],
    static_decoder=None,
)

# --- Build model --------------------------------------------------------------------------------

dqn = build_agent_model(all_configs, env)

# --- Actual search part ------------------------------------------------------------------------------

# 0. Define the environment and noise model
num_transitions = 50 # number of syndrome volumes we sample at each step
num_attacks = 1000 # how often we will try an attack
cost_threshold = 1. # only accept errors that have probability smaller equal than the threshold

visualize = False # plot surface code in each round
# we collect the following statistics if true:
# - Length of an attack path
# - Cost of an attack path
# - Q-Values of incumbent on path
# - Qubit heatmap (total, split by X-,Z-error)
# - Qubit heatmap for first <collect_statistics_steps>
collect_statistics = True
collect_statistics_steps = 5 

if collect_statistics:
    qubit_heatmap = np.zeros((env.d, env.d), int)
    qubit_heatmap_X = np.zeros((env.d, env.d), int)
    qubit_heatmap_Z = np.zeros((env.d, env.d), int)
    path_info = {
        "lengths": []
    }
    qubit_step_heatmaps = []
    for i in range(collect_statistics_steps):
        qubit_step_heatmap = np.zeros((env.d, env.d), int)
        qubit_step_heatmap_X = np.zeros((env.d, env.d), int)
        qubit_step_heatmap_Z = np.zeros((env.d, env.d), int)
        qubit_step_heatmaps.append(
            {
                "all": qubit_step_heatmap,
                "X": qubit_step_heatmap_X,
                "Z": qubit_step_heatmap_Z
            }
        )

for attack in trange(num_attacks):

    # state from which we start local search (start from clean state)
    hidden_state = np.zeros((env.d, env.d), int)
    episode_done = False # indicates if referee can correctly decode state
    transitions = 0 # number of transitions we took

    while not episode_done and transitions < max_transitions:
        # 1. sample a bunch of transitions
        faulty_syndromes = []
        while(len(faulty_syndromes) != num_transitions):
            # For every transition we sample a dxd syndrome volume
            transition_volume = sample_syndrome_volume(env, hidden_state)
            if transition_volume["cost"] <= cost_threshold:
                faulty_syndromes.append(transition_volume)

        # 2. compute q-value of state after transitions
        transition_score_values = []
        for idx in range(num_transitions):
            # Intialize a zero'd input volume
            input_state = np.zeros(
                (1, env.d + env.n_action_layers, 2 * env.d + 1, 2 * env.d + 1), int)
            transition_faulty_syndromes = faulty_syndromes[idx]["syndromes"]
            # embed and place the faulty syndrome slices in the correct place
            for j in range(env.d):
                input_state[:, j, :, :] = env.padding_syndrome(
                    transition_faulty_syndromes[j])
            # get q-values for input_state
            intput_state = dqn.memory.get_recent_state(input_state)
            q_values = dqn.compute_q_values(input_state)
            transition_score_values.append(np.max(q_values))

        # 3. compute score function or directly select one transition
        # select transition according to min(max(q_values)) since we know that agent will act greedy
        incumbent_transition = np.argmin(transition_score_values)
        # print(f"Q-Value score: {transition_score_values[incumbent_transition]}")
        # print(f"Incumbent transition has cost: {faulty_syndromes[incumbent_transition]['cost']}")

        # Intialize a zero'd input volume
        input_state = np.zeros(
            (env.d + env.n_action_layers, 2 * env.d + 1, 2 * env.d + 1), int)
        transition_faulty_syndromes = faulty_syndromes[incumbent_transition]["syndromes"]
        transition_errors = faulty_syndromes[incumbent_transition]["errors"]
        # embed and place the faulty syndrome slices in the correct place
        for j in range(env.d):
            input_state[j, :, :] = env.padding_syndrome(
                transition_faulty_syndromes[j])
        if collect_statistics:
            # update heatmaps
            qubit_heatmap += np.sum(np.array(transition_errors) > 0, axis=0)
            qubit_heatmap_X += np.sum(np.array(transition_errors) == 1, axis=0)
            qubit_heatmap_X += np.sum(np.array(transition_errors) == 2, axis=0)
            qubit_heatmap_Z += np.sum(np.array(transition_errors) == 3, axis=0)
            qubit_heatmap_Z += np.sum(np.array(transition_errors) == 2, axis=0)
            if transitions < collect_statistics_steps:
                qubit_step_heatmaps[transitions]["all"] += np.sum(np.array(transition_errors) > 0, axis=0)
                qubit_step_heatmaps[transitions]["X"] += np.sum(np.array(transition_errors) == 1, axis=0)
                qubit_step_heatmaps[transitions]["X"] += np.sum(np.array(transition_errors) == 2, axis=0)
                qubit_step_heatmaps[transitions]["Z"] += np.sum(np.array(transition_errors) == 3, axis=0)
                qubit_step_heatmaps[transitions]["Z"] += np.sum(np.array(transition_errors) == 2, axis=0)
            # import pdb; pdb.set_trace()

        # 4. Let agent correct until it requests new syndrome volume
        corrections = []
        still_decoding = True
        while still_decoding:
            # Fetch the suggested correction
            action = dqn.forward(input_state)
            if action not in corrections and action != env.identity_index:
                # If the action has not yet been done, or is not the identity
                # append the suggested correction to the list of corrections
                corrections.append(action)
                # Update the input state to the agent to indicate the correction it would have made
                input_state[env.d, :, :] = env.padding_actions(corrections)
            else:
                # decoding should stop
                still_decoding = False
        # print(f"Correcting qubit indices: {corrections}")

        # 5. Update actual hidden state
        # First, we add the error that was generated by the incumbent transition
        for j in range(env.d):
            hidden_state = obtain_new_error_configuration(
                hidden_state, transition_errors[j])
        if visualize:
            # For debugging purposes: Visualize state after corrections
            current_true_syndrome = generate_surface_code_syndrome_NoFT_efficient(hidden_state, env.qubits)
            draw_surface_code(hidden_state, env.syndromes, current_true_syndrome, env.d)

        # Next, we add the corrections of the reinforcement learning agent
        for correction in corrections:
            action_lattice = index_to_move(
                env.d, correction, env.error_model, env.use_Y)
            hidden_state = obtain_new_error_configuration(
                hidden_state, action_lattice)

        # 6. Check if episode is over
        # Call MWPM or NN referee and check homological label
        _, _, episode_done = env.get_reward(hidden_state)

        if visualize:
            # For debugging purposes: Visualize state after corrections
            current_true_syndrome = generate_surface_code_syndrome_NoFT_efficient(hidden_state, env.qubits)
            draw_surface_code(hidden_state, env.syndromes, current_true_syndrome, env.d)
            plt.show()

        transitions += 1

    if(transitions >= max_transitions):
        # print(f"Reached maximum number of transitions: {transitions}/{max_transitions}: no attack found!")
        path_info["lengths"].append(max_transitions)
    else:
        # print(f"Found an attack in {transitions} transitions!")
        path_info["lengths"].append(transitions)

# Plot aggregated qubit heatmaps (total, x-/z-error)
fig_total, (ax1,ax2,ax3) = plt.subplots(1,3)
ax1.matshow(qubit_heatmap)
ax1.set_title("All errors")
ax2.matshow(qubit_heatmap_X)
ax2.set_title("X-errors")
ax3.matshow(qubit_heatmap_Z)
ax3.set_title("Z-errors")
fig_total.suptitle("Aggregated error distribution over Qubits over multiple steps")
plt.tight_layout()
plt.savefig('heatmap_experiments/aggregated_heatmaps.png', dpi=200)

# Plot step heatmaps
fig_steps, ax_steps = plt.subplots(collect_statistics_steps,3, sharey=True)
for i in range(collect_statistics_steps):
    ax_steps[i,0].matshow(qubit_step_heatmaps[i]["all"])
    ax_steps[i,0].set_ylabel(f"Step {i}")
    ax_steps[i,1].matshow(qubit_step_heatmaps[i]["X"])
    ax_steps[i,2].matshow(qubit_step_heatmaps[i]["Z"])
for ax in ax_steps.flat:
    ax.label_outer()
ax_steps[0,0].set_title("All errors")
ax_steps[0,1].set_title("X-errors")
ax_steps[0,2].set_title("Z-errors")
fig_steps.suptitle("Error distribution over Qubits for each step")
plt.tight_layout()
plt.savefig('heatmap_experiments/step_heatmaps.png', dpi=200)

# Now dump heatmaps to pickle file
with open('heatmap_experiments/heatmaps.p', 'wb') as fd:
    data = {
        "agent": trained_model_dir+"/"+error_rate,
        "distance": env.error_model,
        "error_model": env.d,
        "attacks": num_attacks,
        "average_path_length": np.mean(path_info['lengths']),
        "step_heatmaps": qubit_step_heatmaps,
        "heatmap": qubit_heatmap,
        "x_heatmap": qubit_heatmap_X,
        "z_heatmap": qubit_heatmap_Z
    }
    pickle.dump(data, fd, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Average path lenght: {np.mean(path_info['lengths'])}")