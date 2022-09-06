from deepq.Environments import *
from deepq.Function_Library import *
from deepq.Utils import *

import matplotlib.pyplot as plt

"""
Small test to check if MWPM suggest similar corrections as the NN referee decoder
"""


# environment parameter
d=5
error_model='X'
static_decoder_path = f"../example_notebooks/referee_decoders/nn_d5_{error_model}_p5"
static_decoder = load_model(static_decoder_path)

# create the environment
env = Surface_Code_Environment_Multi_Decoding_Cycles(
    d=d, 
    p_phys=0.03, 
    p_meas=0.03,  
    error_model=error_model, 
    use_Y=False, 
    volume_depth=d,
    static_decoder=static_decoder)

# reset environment and generate first syndrome error volume
obs = env.reset()

# instantiate MWPM decoder (MatchingDecoder class)
stab_list = env.get_stabilizer_list(env.qubits, env.d)
H = get_parity_matrix(stab_list, env.syndromes, 3, env.d)
print(H)
matching_decoder = MatchingDecoder(H)

# draw syndromes and flipped data qubits
draw_surface_code(env.hidden_state, env.syndromes, env.current_true_syndrome, env.d)
print(env.hidden_state)

# perform MWPM decoding. Vector z corresponds to data qubit corrections
syn_vec = get_syndrome_vector(env.current_true_syndrome, env.syndromes, 3, env.d)
z = matching_decoder.predict(syn_vec)

# combine corrections with current data qubit
new_hidden_state = obtain_new_error_configuration(env.hidden_state, z.reshape((env.d, env.d)))
# compute syndromes from new state
new_true_syndromes = generate_surface_code_syndrome_NoFT_efficient(new_hidden_state, env.qubits)
# draw corrected surface code
draw_surface_code(new_hidden_state, env.syndromes, new_true_syndromes, env.d)
plt.show()
# get logical qubit labels for current, corrected and referee prediction
l1 = generate_one_hot_labels_surface_code(new_hidden_state, 'X')
l2 = generate_one_hot_labels_surface_code(env.hidden_state, 'X')
vec = np.reshape(env.current_true_syndrome,(env.d+1)**2)
l3 = env.static_decoder.predict(np.array([vec]), batch_size=1, verbose=0)

print(f"state without correction: {l2}")
print(f"state after MWPM: {l1}")
print(f"predicted state: {l3[0]}")