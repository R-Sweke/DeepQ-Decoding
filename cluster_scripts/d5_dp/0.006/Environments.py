# --------------------------------------------------------------------------------------------------------------------------------
#
#   This file contains various surface code environments:
#
#       (1) Surface_Code_Environment_No_Syndrome_History_Fixed_Number_of_Moves_Ground_State_Reward
#       (2) Surface_Code_Environment_With_Syndrome_History_Fixed_Number_of_Moves_Ground_State_Reward
#       (3) Surface_Code_Environment_No_Syndrome_History_Variable_Number_of_Moves_Ground_State_Reward
#       (4) Surface_Code_Environment_With_Syndrome_History_Variable_Number_of_Moves_Ground_State_Reward
#       (5) Surface_Code_Environment_With_Syndrome_Volume_Variable_Number_of_Moves_Ground_State_Reward
#
# These environments are streamlined (no possibility for visual debugging) versions of the environments in Environments_with_debug.py
#
# --------------------------------------------------------------------------------------------------------------------------------

#----- (0) Imports ---------------------------------------------------------------------------------------------------------------

from Function_Library import *
import gym
import copy
from itertools import product, starmap

#---------- (1) --------------------------------------------------------------------------------------------------------------------------------------


class Surface_Code_Environment_No_Syndrome_History_Fixed_Number_of_Moves_Ground_State_Reward():
    """
    This environment has:

        - no syndrome history - i.e. the agent see's only a single syndrome
        - a fixed number of moves between each error/syndrome
        - a plus 1 reward for every action that results in the code being in the ground state
        - can deal with error_model in {"X","DP"}
        - can deal with faulty syndromes - i.e. p_meas > 0

    In particular, after each step the agent sees the most recent syndrome along with a list of actions it has taken since that syndrome was received.

    As in all gym environments, this class contains both a step method, and a reset method.

    Parameters
    ----------

    d: The code distance
    p_phys: The physical error rate
    p_meas: The measurement error rate
    max_moves: The number of moves the agent is allowed before a new error occurs and a new syndrome is received
    debug: These environments do not cater for visual debugging
    decoder_path: A path to a trained referee decoder
    error_model: Should be in ["DP", "X"] - the error model that is being utilized
    use_Y: If true then the agent is allowed to perform Y Paulis, if False then the agent is only allowed to perform X and Z Paulis

    """


    def __init__(self, d=5, p_phys=0.01, p_meas=0.01, max_moves=3, debug=False, decoder_path=None, error_model="DP", use_Y=True):

        self.d = d
        self.p_phys = p_phys
        self.p_meas= p_meas
        self.error_model = error_model
        self.use_Y = use_Y

        if error_model == "X":
            self.num_actions = d**2 + 1
        elif error_model == "DP":
            if use_Y:
                self.num_actions = 3*d**2 + 1
            else:
               self.num_actions = 2*d**2 + 1 
        else:
            print("specified error model not currently supported!")

        self.max_moves = max_moves
        self.qubits = generateSurfaceCodeLattice(d)
        self.observation_space = gym.spaces.MultiBinary((d+1)**2 + self.num_actions)
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.hidden_state = np.zeros((self.d,self.d))                  
        self.invisible_syndrome = np.zeros((self.d+1)**2)               
        self.visible_syndrome_post_error = np.zeros((self.d+1)**2)
        self.completed_actions = np.zeros(self.num_actions)             
        self.completed_action_count = 0                           
        self.state = np.concatenate([self.visible_syndrome_post_error,self.completed_actions])
        self.static_decoder = None
        self.debug=debug
        self.has_loaded_decoder = False
        self.decoder_path = decoder_path

    def step(self, action):

        # 0) Check that the decoder model has been loaded, if not then load it...
        if not self.has_loaded_decoder:
            self.static_decoder = keras.models.load_model(self.decoder_path)
            self.has_loaded_decoder = True

        new_error_flag = False

        # 1) Apply the action to the hidden state
        action_lattice = index_to_move(self.d,action,self.error_model,self.use_Y)
        self.hidden_state = obtain_new_error_configuration(self.hidden_state, action_lattice)
        self.completed_action_count +=1
        

        # 2) Update the invisible syndrome (i.e. what the static decoder sees)
        invisible_syndrome_lattice = generate_surface_code_syndrome_NoFT_efficient(self.hidden_state,self.qubits)
        self.invisible_syndrome = np.reshape(invisible_syndrome_lattice,(self.d+1)**2)

        # 2) Calculate the reward
        num_anyons = np.sum(self.invisible_syndrome)
        correct_label = generate_one_hot_labels_surface_code(self.hidden_state,self.error_model)
        decoder_label = self.static_decoder.predict(np.array([self.invisible_syndrome]), batch_size=1, verbose=0)

        done = False
        reward = 0

        if np.argmax(correct_label) == 0 and num_anyons == 0:
            reward = 1.0
        elif np.argmax(decoder_label[0]) != np.argmax(correct_label):
            done = True

        # 3) If necessary, apply an error
        if self.completed_action_count == self.max_moves:

            # First apply the error
            error = generate_error(self.d,self.p_phys,self.error_model)
            self.hidden_state = obtain_new_error_configuration(self.hidden_state,error)
            new_error_flag = True

            # Update the syndromes 
            syndrome_lattice = generate_surface_code_syndrome_NoFT_efficient(self.hidden_state,self.qubits)
            faulty_syndrome_lattice = generate_faulty_syndrome(syndrome_lattice,self.p_meas)
            self.invisible_syndrome = np.reshape(syndrome_lattice,(self.d+1)**2)
            self.visible_syndrome_post_error = np.reshape(faulty_syndrome_lattice,(self.d+1)**2)
            
            # reset the completed actions
            self.completed_actions = np.zeros(self.num_actions)
            self.completed_action_count = 0

        else:
            # Update the completed actions, but not the visible syndrome
            self.completed_actions[action] = float(not(self.completed_actions[action]))
            
        # update the combined state
        self.state = np.concatenate([self.visible_syndrome_post_error,self.completed_actions])

        return self.state, reward, done, {"new_error_flag": new_error_flag}


    def reset(self):
        self.hidden_state = np.zeros((self.d,self.d))                        
        self.invisible_syndrome = np.zeros((self.d+1)**2)              
        self.visible_syndrome_post_error = np.zeros((self.d+1)**2)
        self.completed_actions = np.zeros(self.num_actions)            
        self.state = np.concatenate([self.visible_syndrome_post_error,self.completed_actions])
        self.completed_action_count = 0   
        return self.state

# --------- (2)  -------------------------------------------------------------------------------------------------------------------------------------


class Surface_Code_Environment_With_Syndrome_History_Fixed_Number_of_Moves_Ground_State_Reward():
    """
    This environment has:

        - syndrome history - i.e. the agent see's not just the most recent syndrome but a user specified number of most recent syndromes
        - a fixed number of moves between each error/syndrome
        - a plus 1 reward for every action that results in the code being in the ground state
        - can deal with error_model in {"X","DP"}
        - can deal with faulty syndromes - i.e. p_meas > 0

    In particular, after each step the agent sees the most recent "syndrome_memory length" number of syndromes along with a list of actions it took between each syndrome.

    As in all gym environments, this class contains both a step method, and a reset method.

    Parameters
    ----------

    d: The code distance
    p_phys: The physical error rate
    p_meas: The measurement error rate
    max_moves: The number of moves the agent is allowed before a new error occurs and a new syndrome is received
    debug: These environments do not cater for visual debugging
    decoder_path: A path to a trained referee decoder
    error_model: Should be in ["DP", "X"] - the error model that is being utilized
    use_Y: If true then the agent is allowed to perform Y Paulis, if False then the agent is only allowed to perform X and Z Paulis
    syndrome_memory_length: This specifies the number of most recent syndromes seen by the agent
    flatten_state: If true, then the state returned by the step method is flattened into a vector which is suitable for input into a feed forward neural network

    """

    def __init__(self, d=5, p_phys=0.01, p_meas=0.01, max_moves=3, debug=False, decoder_path=None, error_model="X", use_Y=True, syndrome_memory_length=3, flatten_state = False):

        self.d = d
        self.p_phys = p_phys
        self.p_meas= p_meas
        self.error_model = error_model
        self.use_Y = use_Y
        self.flatten_state = flatten_state

        if error_model == "X":
            self.num_actions = d**2 + 1
        elif error_model == "DP":
            if use_Y:
                self.num_actions = 3*d**2 + 1
            else:
               self.num_actions = 2*d**2 + 1 
        else:
            print("specified error model not currently supported!")
            
        self.syndrome_size = (d+1)**2
        self.num_features = self.syndrome_size + self.num_actions 
        self.syndrome_memory_length = syndrome_memory_length
        self.max_moves = max_moves
        self.qubits = generateSurfaceCodeLattice(d)

        if not self.flatten_state:
            self.observation_space = gym.spaces.Box(low=0,high=1,shape=(self.syndrome_memory_length,self.num_features),dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.MultiBinary(self.syndrome_memory_length*self.num_features)

        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.hidden_state = np.zeros((self.d,self.d))        
        
        self.current_invisible_syndrome = np.zeros(self.syndrome_size)               
        self.current_visible_syndrome_post_error = np.zeros(self.syndrome_size)
        self.current_completed_actions = np.zeros(self.num_actions)
        self.current_completed_action_count = 0
        
        self.visible_state_memory = [[np.zeros(self.syndrome_size),np.zeros(self.num_actions)] for j in range(self.syndrome_memory_length)]
        self.total_state = np.zeros((self.syndrome_memory_length, self.num_features))
        self.flattened_state = np.reshape(self.total_state, self.syndrome_memory_length*self.num_features)

        self.debug = debug
        self.static_decoder = None
        self.has_loaded_decoder = False
        self.decoder_path = decoder_path

        

    def step(self, action):

        # 0) Check that the decoder model has been loaded, if not then load it...
        if not self.has_loaded_decoder:
            self.static_decoder = keras.models.load_model(self.decoder_path)
            self.has_loaded_decoder = True

        new_error_flag = False
        # 1) Apply the action to the hidden state
        action_lattice = index_to_move(self.d,action,self.error_model,self.use_Y)
        self.hidden_state = obtain_new_error_configuration(self.hidden_state, action_lattice)
        self.current_completed_action_count += 1

        # 2) Update the current invisible syndrome (i.e. what the static decoder sees)
        invisible_syndrome_lattice = generate_surface_code_syndrome_NoFT_efficient(self.hidden_state,self.qubits)
        self.current_invisible_syndrome = np.reshape(invisible_syndrome_lattice,(self.d+1)**2)

        # 2) Calculate the reward
        num_anyons = np.sum(self.current_invisible_syndrome)
        correct_label = generate_one_hot_labels_surface_code(self.hidden_state,self.error_model)
        decoder_label = self.static_decoder.predict(np.array([self.current_invisible_syndrome]), batch_size=1, verbose=0)

        done = False
        reward = 0

        if np.argmax(correct_label) == 0 and num_anyons == 0:
            reward = 1.0
        elif np.argmax(decoder_label[0]) != np.argmax(correct_label):
            done = True

        # 3) If necessary, apply an error
        if self.current_completed_action_count == self.max_moves:
            
            # Update the flag
            new_error_flag=True
            
            # If an error has been applied, we need to update the action count, then shift back in the memory
            self.current_completed_actions[action] = float(not(self.current_completed_actions[action]))
            
            if self.syndrome_memory_length > 1:
                memory_copy = copy.copy(self.visible_state_memory)
                for j in range(1,self.syndrome_memory_length):
                    self.visible_state_memory[j] = memory_copy[j-1]         #NB: 0 is the most recent!!
            
            # Now we can apply the error
            error = generate_error(self.d,self.p_phys,self.error_model)
            self.hidden_state = obtain_new_error_configuration(self.hidden_state,error)
            
            # Update the syndromes...
            syndrome_lattice = generate_surface_code_syndrome_NoFT_efficient(self.hidden_state,self.qubits)
            faulty_syndrome_lattice = generate_faulty_syndrome(syndrome_lattice,self.p_meas)
            self.current_invisible_syndrome = np.reshape(syndrome_lattice,(self.d+1)**2)
            self.current_visible_syndrome_post_error = np.reshape(faulty_syndrome_lattice,(self.d+1)**2)

            # reset the completed actions
            self.current_completed_actions = np.zeros(self.num_actions)
            self.current_completed_action_count = 0

        else:
            # Update the completed actions, but not the visible syndrome
            self.current_completed_actions[action] = float(not(self.current_completed_actions[action]))
            
        # insert the latest syndrome and action into the memory
        self.visible_state_memory[0] = [self.current_visible_syndrome_post_error, self.current_completed_actions]
        
        # Update the total state visible to the model (this can definitely be done more efficiently!)                      
        for j in range(self.syndrome_memory_length):
            self.total_state[j,:] = np.concatenate([self.visible_state_memory[j][0],self.visible_state_memory[j][1]])

        if self.flatten_state:
            self.flattened_state = np.reshape(self.total_state, self.syndrome_memory_length*self.num_features)
            return self.flattened_state, reward, done, new_error_flag
        else:
            return self.total_state, reward, done, {"new_error_flag": new_error_flag}


    def reset(self):
        
        self.hidden_state = np.zeros((self.d,self.d))        
        self.current_invisible_syndrome = np.zeros(self.syndrome_size)               
        self.current_visible_syndrome_post_error = np.zeros(self.syndrome_size)
        self.current_completed_actions = np.zeros(self.num_actions)
        self.current_completed_action_count = 0
        self.visible_state_memory = [[np.zeros(self.syndrome_size),np.zeros(self.num_actions)] for j in range(self.syndrome_memory_length)]
        self.total_state = np.zeros((self.syndrome_memory_length, self.num_features)) 
        self.flattened_state = np.reshape(self.total_state, self.syndrome_memory_length*self.num_features)
        
        if self.flatten_state:
            return self.flattened_state
        else:
            return self.total_state

#---------- (3) --------------------------------------------------------------------------------------------------------------------------------------


class Surface_Code_Environment_No_Syndrome_History_Variable_Number_of_Moves_Ground_State_Reward():
    """
    This environment has:

        - no syndrome history - i.e. the agent see's only a single syndrome
        - a _variable_ number of moves - i.e. a new error is only introduced when the agent performs an identity (which can be done by performing the same Pauli twice)
        - a plus 1 reward for every action that results in the code being in the ground state
        - can deal with error_model in {"X","DP"}
        - can deal with faulty syndromes - i.e. p_meas > 0

    In particular, after each step the agent sees the most recent syndrome along with a list of actions it has taken since that syndrome was received. After doing an identity, a
    new error occurs and a new syndrome is received

    As in all gym environments, this class contains both a step method, and a reset method.

    Parameters
    ----------

    d: The code distance
    p_phys: The physical error rate
    p_meas: The measurement error rate
    max_moves: The number of moves the agent is allowed before a new error occurs and a new syndrome is received
    debug: These environments do not cater for visual debugging
    decoder_path: A path to a trained referee decoder
    error_model: Should be in ["DP", "X"] - the error model that is being utilized
    use_Y: If true then the agent is allowed to perform Y Paulis, if False then the agent is only allowed to perform X and Z Paulis

    """


    def __init__(self, d=5, p_phys=0.01, p_meas=0.01, debug=False, decoder_path=None, error_model="DP", use_Y=True):

        self.d = d
        self.p_phys = p_phys
        self.p_meas= p_meas
        self.error_model = error_model
        self.use_Y = use_Y

        if error_model == "X":
            self.num_actions = d**2 + 1
        elif error_model == "DP":
            if use_Y:
                self.num_actions = 3*d**2 + 1
            else:
               self.num_actions = 2*d**2 + 1 
        else:
            print("specified error model not currently supported!")

        self.identity_index = self.num_actions -1
        self.qubits = generateSurfaceCodeLattice(d)
        self.observation_space = gym.spaces.MultiBinary((d+1)**2 + self.num_actions)
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.hidden_state = np.zeros((self.d,self.d))                  
        self.invisible_syndrome = np.zeros((self.d+1)**2)               
        self.visible_syndrome_post_error = np.zeros((self.d+1)**2)
        self.completed_actions = np.zeros(self.num_actions)             
        self.completed_action_count = 0                           
        self.state = np.concatenate([self.visible_syndrome_post_error,self.completed_actions])
        self.static_decoder = None
        self.debug=debug
        self.has_loaded_decoder = False
        self.decoder_path = decoder_path

    def step(self, action):

        # 0) Check that the decoder model has been loaded, if not then load it...
        if not self.has_loaded_decoder:
            self.static_decoder = keras.models.load_model(self.decoder_path)
            self.has_loaded_decoder = True

        new_error_flag = False

        # 1) Apply the action to the hidden state
        action_lattice = index_to_move(self.d,action,self.error_model,self.use_Y)
        self.hidden_state = obtain_new_error_configuration(self.hidden_state, action_lattice)
        self.completed_action_count +=1

        # 2) Update the invisible syndrome (i.e. what the static decoder sees)
        invisible_syndrome_lattice = generate_surface_code_syndrome_NoFT_efficient(self.hidden_state,self.qubits)
        self.invisible_syndrome = np.reshape(invisible_syndrome_lattice,(self.d+1)**2)

        # 2) Calculate the reward
        num_anyons = np.sum(self.invisible_syndrome)
        correct_label = generate_one_hot_labels_surface_code(self.hidden_state,self.error_model)
        decoder_label = self.static_decoder.predict(np.array([self.invisible_syndrome]), batch_size=1, verbose=0)

        done = False
        reward = 0

        if np.argmax(correct_label) == 0 and num_anyons == 0:
            reward = 1.0
        elif np.argmax(decoder_label[0]) != np.argmax(correct_label):
            done = True

        # 3) If necessary, apply an error
        if action == self.identity_index or float(self.completed_actions[action]) == 1.0:

            # First apply the error
            error = generate_error(self.d,self.p_phys,self.error_model)
            self.hidden_state = obtain_new_error_configuration(self.hidden_state,error)
            new_error_flag = True


            # Update the syndromes 
            syndrome_lattice = generate_surface_code_syndrome_NoFT_efficient(self.hidden_state,self.qubits)
            faulty_syndrome_lattice = generate_faulty_syndrome(syndrome_lattice,self.p_meas)
            self.invisible_syndrome = np.reshape(syndrome_lattice,(self.d+1)**2)
            self.visible_syndrome_post_error = np.reshape(faulty_syndrome_lattice,(self.d+1)**2)
            
            # reset the completed actions
            self.completed_actions = np.zeros(self.num_actions)
            self.completed_action_count = 0

        else:
            # Update the completed actions, but not the visible syndrome
            self.completed_actions[action] = float(not(self.completed_actions[action]))
            
        # update the combined state
        self.state = np.concatenate([self.visible_syndrome_post_error,self.completed_actions])

        return self.state, reward, done, {"new_error_flag": new_error_flag}


    def reset(self):
        self.hidden_state = np.zeros((self.d,self.d))                        
        self.invisible_syndrome = np.zeros((self.d+1)**2)              
        self.visible_syndrome_post_error = np.zeros((self.d+1)**2)
        self.completed_actions = np.zeros(self.num_actions)            
        self.state = np.concatenate([self.visible_syndrome_post_error,self.completed_actions])
        self.completed_action_count = 0   
        return self.state


# --------- (4)  -------------------------------------------------------------------------------------------------------------------------------------


class Surface_Code_Environment_With_Syndrome_History_Variable_Number_of_Moves_Ground_State_Reward():
    """
    This environment has:

        - syndrome history - i.e. the agent see's not just the most recent syndrome but a user specified number of most recent syndromes
        - a _variable_ number of moves - i.e. a new error is only introduced when the agent performs an identity (which can be done by performing the same Pauli twice)
        - a plus 1 reward for every action that results in the code being in the ground state
        - can deal with error_model in {"X","DP"}
        - can deal with faulty syndromes - i.e. p_meas > 0

    In particular, after each step the agent sees the most recent "syndrome_memory length" number of syndromes along with a list of actions it took between each syndrome.

    As in all gym environments, this class contains both a step method, and a reset method.

    Parameters
    ----------

    d: The code distance
    p_phys: The physical error rate
    p_meas: The measurement error rate
    max_moves: The number of moves the agent is allowed before a new error occurs and a new syndrome is received
    debug: Does not cater for visual debugging
    decoder_path: A path to a trained referee decoder
    error_model: Should be in ["DP", "X"] - the error model that is being utilized
    use_Y: If true then the agent is allowed to perform Y Paulis, if False then the agent is only allowed to perform X and Z Paulis
    syndrome_memory_length: This specifies the number of most recent syndromes seen by the agent
    flatten_state: If true, then the state returned by the step method is flattened into a vector which is suitable for input into a feed forward neural network

    """


    def __init__(self, d=5, p_phys=0.01, p_meas=0.01, debug=False, decoder_path=None, error_model="X", use_Y=True, syndrome_memory_length=3, flatten_state=False):

        self.d = d
        self.p_phys = p_phys
        self.p_meas= p_meas
        self.error_model = error_model
        self.use_Y = use_Y
        self.flatten_state = flatten_state

        if error_model == "X":
            self.num_actions = d**2 + 1
        elif error_model == "DP":
            if use_Y:
                self.num_actions = 3*d**2 + 1
            else:
               self.num_actions = 2*d**2 + 1 
        else:
            print("specified error model not currently supported!")
        
        self.identity_index = self.num_actions -1
        self.syndrome_size = (d+1)**2
        self.num_features = self.syndrome_size + self.num_actions 
        self.syndrome_memory_length = syndrome_memory_length
        self.qubits = generateSurfaceCodeLattice(d)

        if not self.flatten_state:
            self.observation_space = gym.spaces.Box(low=0,high=1,shape=(self.syndrome_memory_length,self.num_features),dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.MultiBinary(self.syndrome_memory_length*self.num_features)

        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.hidden_state = np.zeros((self.d,self.d))        
        
        self.current_invisible_syndrome = np.zeros(self.syndrome_size)               
        self.current_visible_syndrome_post_error = np.zeros(self.syndrome_size)
        self.current_completed_actions = np.zeros(self.num_actions)
        self.current_completed_action_count = 0
        
        self.visible_state_memory = [[np.zeros(self.syndrome_size),np.zeros(self.num_actions)] for j in range(self.syndrome_memory_length)]
        self.total_state = np.zeros((self.syndrome_memory_length, self.num_features))
        self.flattened_state = np.reshape(self.total_state, self.syndrome_memory_length*self.num_features)
        
        self.debug = debug
        self.static_decoder = None
        self.has_loaded_decoder = False
        self.decoder_path = decoder_path
        

    def step(self, action):

        # 0) Check that the decoder model has been loaded, if not then load it...
        if not self.has_loaded_decoder:
            self.static_decoder = keras.models.load_model(self.decoder_path)
            self.has_loaded_decoder = True

        new_error_flag = False
        # 1) Apply the action to the hidden state
        action_lattice = index_to_move(self.d,action,self.error_model,self.use_Y)
        self.hidden_state = obtain_new_error_configuration(self.hidden_state, action_lattice)
        self.current_completed_action_count += 1

        # 2) Update the current invisible syndrome (i.e. what the static decoder sees)
        invisible_syndrome_lattice = generate_surface_code_syndrome_NoFT_efficient(self.hidden_state,self.qubits)
        self.current_invisible_syndrome = np.reshape(invisible_syndrome_lattice,(self.d+1)**2)

        # 2) Calculate the reward
        num_anyons = np.sum(self.current_invisible_syndrome)
        correct_label = generate_one_hot_labels_surface_code(self.hidden_state,self.error_model)
        decoder_label = self.static_decoder.predict(np.array([self.current_invisible_syndrome]), batch_size=1, verbose=0)

        done = False
        reward = 0

        if np.argmax(correct_label) == 0 and num_anyons == 0:
            reward = 1.0
        elif np.argmax(decoder_label[0]) != np.argmax(correct_label):
            done = True

        # 3) If necessary, apply an error
        if action == self.identity_index or float(self.current_completed_actions[action]) == 1.0:
            
            # Update the flag
            new_error_flag=True
            
            # If an error has been applied, we need to update the action count, then shift back in the memory
            self.current_completed_actions[action] = float(not(self.current_completed_actions[action]))
            
            if self.syndrome_memory_length > 1:
                memory_copy = copy.copy(self.visible_state_memory)
                for j in range(1,self.syndrome_memory_length):
                    self.visible_state_memory[j] = memory_copy[j-1]         #NB: 0 is the most recent!!
            
            # Now we can apply the error
            error = generate_error(self.d,self.p_phys,self.error_model)
            self.hidden_state = obtain_new_error_configuration(self.hidden_state,error)

            # Update the syndromes...
            syndrome_lattice = generate_surface_code_syndrome_NoFT_efficient(self.hidden_state,self.qubits)
            faulty_syndrome_lattice = generate_faulty_syndrome(syndrome_lattice,self.p_meas)
            self.current_invisible_syndrome = np.reshape(syndrome_lattice,(self.d+1)**2)
            self.current_visible_syndrome_post_error = np.reshape(faulty_syndrome_lattice,(self.d+1)**2)

            # reset the completed actions
            self.current_completed_actions = np.zeros(self.num_actions)
            self.current_completed_action_count = 0

        else:
            # Update the completed actions, but not the visible syndrome
            self.current_completed_actions[action] = float(not(self.current_completed_actions[action]))

        # insert the latest syndrome and action into the memory
        self.visible_state_memory[0] = [self.current_visible_syndrome_post_error, self.current_completed_actions]
        
        # Update the total state visible to the model (this can definitely be done more efficiently!)                      
        for j in range(self.syndrome_memory_length):
            self.total_state[j,:] = np.concatenate([self.visible_state_memory[j][0],self.visible_state_memory[j][1]])

        if self.flatten_state:
            self.flattened_state = np.reshape(self.total_state, self.syndrome_memory_length*self.num_features)
            return self.flattened_state, reward, done, new_error_flag
        else:
            return self.total_state, reward, done, {"new_error_flag": new_error_flag}

    def reset(self):
        
        self.hidden_state = np.zeros((self.d,self.d))        
        self.current_invisible_syndrome = np.zeros(self.syndrome_size)               
        self.current_visible_syndrome_post_error = np.zeros(self.syndrome_size)
        self.current_completed_actions = np.zeros(self.num_actions)
        self.current_completed_action_count = 0
        self.visible_state_memory = [[np.zeros(self.syndrome_size),np.zeros(self.num_actions)] for j in range(self.syndrome_memory_length)]
        self.total_state = np.zeros((self.syndrome_memory_length, self.num_features))
        self.flattened_state = np.reshape(self.total_state, self.syndrome_memory_length*self.num_features) 
        
        if self.flatten_state:
            return self.flattened_state
        else:
            return self.total_state


# --------- (5) ----------------------------------------------------------------------------------------------------------------------------------------

class Surface_Code_Environment_With_Syndrome_Volume_Variable_Number_of_Moves_Ground_State_Reward():
    """
    This environment has:

        - no syndrome history - i.e. the agent see's only a single syndrome
        - a _variable_ number of moves - i.e. a new error is only introduced when the agent performs an identity (which can be done by performing the same Pauli twice)
        - a plus 1 reward for every action that results in the code being in the ground state
        - can deal with error_model in {"X","DP"}
        - can deal with faulty syndromes - i.e. p_meas > 0

    In particular, after each step the agent sees the most recent syndrome along with a list of actions it has taken since that syndrome was received. After doing an identity, a
    new error occurs and a new syndrome is received

    As in all gym environments, this class contains both a step method, and a reset method.

    Parameters
    ----------

    d: The code distance
    p_phys: The physical error rate
    p_meas: The measurement error rate
    max_moves: The number of moves the agent is allowed before a new error occurs and a new syndrome is received
    debug: If true then images of game play will be generated live during calls to the environment
    decoder_path: A path to a trained referee decoder
    error_model: Should be in ["DP", "X"] - the error model that is being utilized
    use_Y: If true then the agent is allowed to perform Y Paulis, if False then the agent is only allowed to perform X and Z Paulis

    """


    def __init__(self, d=5, p_phys=0.01, p_meas=0.01, debug=False, decoder_path=None, error_model="DP", use_Y=True, volume_depth=3, increments=False):

        self.d = d
        self.p_phys = p_phys
        self.p_meas= p_meas
        self.error_model = error_model
        self.use_Y = use_Y
        self.increments = increments

        if error_model == "X":
            self.num_actions = d**2 + 1
        elif error_model == "DP":
            if use_Y:
                self.num_actions = 3*d**2 + 1
            else:
               self.num_actions = 2*d**2 + 1 
        else:
            print("specified error model not currently supported!")

        self.identity_index = self.num_actions -1
        self.qubits = generateSurfaceCodeLattice(d)

        self.volume_depth = volume_depth
        self.syndrome_length = (self.d+1)**2
        self.observation_space = gym.spaces.MultiBinary(self.volume_depth*self.syndrome_length + self.num_actions)
        self.action_space = gym.spaces.Discrete(self.num_actions)

        self.current_hidden_state = np.zeros((self.d,self.d))
        self.volume_hidden_states = [np.zeros((self.d,self.d)) for j in range(self.volume_depth)] 
        self.volume_true_syndromes = [np.zeros((self.d+1,self.d+1)) for j in range(self.volume_depth)]
        self.volume_faulty_syndromes = [np.zeros((self.d+1,self.d+1)) for j in range(self.volume_depth)]
        self.visible_volume = [np.zeros((self.d+1,self.d+1)) for j in range(self.volume_depth)]                          # This may be increments, or faulty syndromes

        self.completed_actions = np.zeros(self.num_actions)             
        self.completed_action_count = 0  

        self.volume_state = np.reshape(self.visible_volume[0],self.syndrome_length)  
        for j in range(1, self.volume_depth):
            self.volume_state = np.concatenate((self.volume_state,np.reshape(self.visible_volume[j],self.syndrome_length))) 

        self.state = np.concatenate((self.volume_state, self.completed_actions))                      

        self.static_decoder = None
        self.debug=debug
        self.has_loaded_decoder = False
        self.decoder_path = decoder_path

    def step(self, action):

        # 0) Check that the decoder model has been loaded, if not then load it...
        if not self.has_loaded_decoder:
            self.static_decoder = keras.models.load_model(self.decoder_path)
            self.has_loaded_decoder = True

        new_error_flag = False

        # 1) Apply the action to the hidden state
        action_lattice = index_to_move(self.d,action,self.error_model,self.use_Y)
        self.current_hidden_state = obtain_new_error_configuration(self.current_hidden_state, action_lattice)
        self.completed_action_count +=1

        # 2) Update the invisible syndrome (i.e. what the static decoder sees)
        current_true_syndrome_lattice = generate_surface_code_syndrome_NoFT_efficient(self.current_hidden_state,self.qubits)
        current_true_syndrome_vector = np.reshape(current_true_syndrome_lattice,self.syndrome_length)

        # 2) Calculate the reward
        num_anyons = np.sum(current_true_syndrome_vector)
        correct_label = generate_one_hot_labels_surface_code(self.current_hidden_state,self.error_model)
        decoder_label = self.static_decoder.predict(np.array([current_true_syndrome_vector]), batch_size=1, verbose=0)

        done = False
        reward = 0

        if np.argmax(correct_label) == 0 and num_anyons == 0:
            reward = 1.0
        elif np.argmax(decoder_label[0]) != np.argmax(correct_label):
            done = True

        # 3) If necessary, apply multiple errors and obtain an error volume
        if action == self.identity_index or float(self.completed_actions[action]) == 1.0:

            # Flag that errors have occured in this step:
            new_error_flag = True

            # Now we want to generate a volume of errors and syndromes
            for j in range(self.volume_depth):
                error = generate_error(self.d,self.p_phys,self.error_model)
                self.current_hidden_state = obtain_new_error_configuration(self.current_hidden_state,error)
                self.volume_hidden_states[j] = copy.copy(self.current_hidden_state)
                self.volume_true_syndromes[j] = generate_surface_code_syndrome_NoFT_efficient(self.current_hidden_state,self.qubits)
                self.volume_faulty_syndromes[j] = generate_faulty_syndrome(self.volume_true_syndromes[j],self.p_meas)


            if not self.increments:
                self.visible_volume = copy.copy(self.volume_faulty_syndromes)
            else:
                self.visible_volume[0] = copy.copy(self.volume_faulty_syndromes[0])
                for j in range(1,self.volume_depth):
                    self.visible_volume[j] = np.abs(self.volume_faulty_syndromes[j] -self.volume_faulty_syndromes[j-1])


            self.volume_state = np.reshape(self.visible_volume[0],self.syndrome_length)  
            for j in range(1, self.volume_depth):
                self.volume_state = np.concatenate((self.volume_state,np.reshape(self.visible_volume[j],self.syndrome_length))) 

            # reset the completed actions
            self.completed_actions = np.zeros(self.num_actions)
            self.completed_action_count = 0

        else:
            # Update the completed actions, but not the visible syndrome
            self.completed_actions[action] = float(not(self.completed_actions[action]))
 
        # update the combined state
        self.state = np.concatenate((self.volume_state, self.completed_actions))    

        return self.state, reward, done, {"new_error_flag": new_error_flag}


    def reset(self):
        self.current_hidden_state = np.zeros((self.d,self.d))
        self.volume_hidden_states = [np.zeros((self.d,self.d)) for j in range(self.volume_depth)] 
        self.volume_true_syndromes = [np.zeros((self.d+1,self.d+1)) for j in range(self.volume_depth)]
        self.volume_faulty_syndromes = [np.zeros((self.d+1,self.d+1)) for j in range(self.volume_depth)]
        self.visible_volume = [np.zeros((self.d+1,self.d+1)) for j in range(self.volume_depth)]                          

        self.completed_actions = np.zeros(self.num_actions)             
        self.completed_action_count = 0  

        self.volume_state = np.reshape(self.visible_volume[0],self.syndrome_length)  
        for j in range(1, self.volume_depth):
            self.volume_state = np.concatenate((self.volume_state,np.reshape(self.visible_volume[j],self.syndrome_length))) 

        self.state = np.concatenate((self.volume_state, self.completed_actions))   
        return self.state

# ------------ (6) ----------------------------------------------------------------------------------------------------------------------

class Surface_Code_Environment_Boxed_Volume_Variable_Number_of_Moves_Ground_State_Reward():
    """
    This environment has:

        - a syndrome volume + completed actions as the state 
        - a variable number of moves:
            - an error is introduced if the agent does the identity
            - an error is introduced if the agent repeats the same move twice
        - a plus 1 reward for every action that results in the code being in the ground state
        - can deal with error_model in {"X","DP"}
        - can deal with faulty syndromes - i.e. p_meas > 0

    TODO: flesh out documentation!

    Parameters
    ----------


    Returns
    -------

    """


    def __init__(self, d=5, p_phys=0.01, p_meas=0.01, debug=False, decoder_path=None, error_model="DP", use_Y=True, volume_depth=3, increments=False, baselines=False):

        self.d = d
        self.p_phys = p_phys
        self.p_meas= p_meas
        self.error_model = error_model
        self.use_Y = use_Y
        self.increments = increments
        self.baselines=baselines

        self.n_action_layers = 0
        if error_model == "X":
            self.num_actions = d**2 + 1
            self.n_action_layers = 1
        elif error_model == "DP":
            if use_Y:
                self.num_actions = 3*d**2 + 1
                self.n_action_layers = 3
            else:
                self.num_actions = 2*d**2 + 1 
                self.n_action_layers = 2
        else:
            print("specified error model not currently supported!")

        self.identity_index = self.num_actions -1
        self.qubits = generateSurfaceCodeLattice(d)

        self.volume_depth = volume_depth
        self.syndrome_length = (self.d+1)**2
        
        
        self.observation_space=gym.spaces.Box(low=0,high=1,
                                      shape=(2*self.d+1, 
                                             2*self.d+1,
                                             self.volume_depth+self.n_action_layers),
                                      dtype=np.uint8)
        
        self.action_space = gym.spaces.Discrete(self.num_actions)

        self.current_hidden_state = np.zeros((self.d,self.d),int)
        self.volume_hidden_states = [np.zeros((self.d,self.d),int) for j in range(self.volume_depth)] 
        self.volume_true_syndromes = [np.zeros((self.d+1,self.d+1),int) for j in range(self.volume_depth)]
        self.volume_faulty_syndromes = [np.zeros((self.d+1,self.d+1),int) for j in range(self.volume_depth)]
        self.visible_volume = [np.zeros((self.d+1,self.d+1),int) for j in range(self.volume_depth)]                          # This may be increments, or faulty syndromes

        self.completed_actions = np.zeros(self.num_actions,int)             
        self.completed_action_count = 0  

        self.state = np.zeros((self.observation_space.shape),int)
        self.base_syndrome = np.zeros((self.d+1,self.d+1),int)
        for j in range(self.volume_depth):
            self.state[:,:,j] = self.padding_syndrome(self.base_syndrome)

        self.static_decoder = None
        self.debug=debug
        self.has_loaded_decoder = False
        self.decoder_path = decoder_path
        
    def padding_syndrome(self, syndrome_in):
        syndrome_out = np.zeros((2*self.d+1, 2*self.d+1),int)
        
        for x in range( 2*self.d+1 ):
            for y in range( 2*self.d+1 ):
                if x%2 == 0 and y%2 == 0: # copy in the syndrome
                    syndrome_out[ x, y ] = syndrome_in[ int(x/2), int(y/2) ]
                elif x%2 == 1 and y%2 == 1:
                    if (x+y)%4 == 0:
                        syndrome_out[x,y] = 1
        return syndrome_out
        
    def padding_actions(self,actions_in):
        # works only for a subset of actions (only x actions)
        # has to be called 1, 2 or 3 separate times, depending on noise-model and use_Y; for X, Z and Y actions
        actions_out = np.zeros( ( 2*self.d+1, 2*self.d+1 ),int )

        for action_index, action_taken in enumerate( actions_in ):
            if action_taken:
                row = int( action_index / self.d )
                col = int( action_index % self.d )

                actions_out[ int( 2*row+1 ), int( 2*col+1 ) ] = 1

        return actions_out

    def step(self, action):

        # 0) Check that the decoder model has been loaded, if not then load it...
        if not self.has_loaded_decoder:
            self.static_decoder = keras.models.load_model(self.decoder_path)
            self.has_loaded_decoder = True

        new_error_flag = False

        # 1) Apply the action to the hidden state
        action_lattice = index_to_move(self.d,action,self.error_model,self.use_Y)
        self.current_hidden_state = obtain_new_error_configuration(self.current_hidden_state, action_lattice)
        self.completed_action_count +=1

        # 2) Update the invisible syndrome (i.e. what the static decoder sees)
        current_true_syndrome_lattice = generate_surface_code_syndrome_NoFT_efficient(self.current_hidden_state,self.qubits)
        current_true_syndrome_vector = np.reshape(current_true_syndrome_lattice,self.syndrome_length)

        # 2) Calculate the reward
        num_anyons = np.sum(current_true_syndrome_vector)
        correct_label = generate_one_hot_labels_surface_code(self.current_hidden_state,self.error_model)
        decoder_label = self.static_decoder.predict(np.array([current_true_syndrome_vector]), batch_size=1, verbose=0)

        done = False
        reward = 0

        if np.argmax(correct_label) == 0 and num_anyons == 0:
            reward = 1.0
        elif np.argmax(decoder_label[0]) != np.argmax(correct_label):
            done = True

        # 3) If necessary, apply multiple errors and obtain an error volume
        if action == self.identity_index or float(self.completed_actions[action]) == 1.0:

            # Flag that errors have occured in this step:
            new_error_flag = True

            # Now we want to generate a volume of errors and syndromes
            for j in range(self.volume_depth):
                error = generate_error(self.d,self.p_phys,self.error_model)
                self.current_hidden_state = obtain_new_error_configuration(self.current_hidden_state,error)
                self.volume_hidden_states[j] = copy.copy(self.current_hidden_state)
                self.volume_true_syndromes[j] = generate_surface_code_syndrome_NoFT_efficient(self.current_hidden_state,self.qubits)
                self.volume_faulty_syndromes[j] = generate_faulty_syndrome(self.volume_true_syndromes[j],self.p_meas)


            if not self.increments:
                self.visible_volume = copy.copy(self.volume_faulty_syndromes)
            else:
                self.visible_volume[0] = copy.copy(self.volume_faulty_syndromes[0])
                for j in range(1,self.volume_depth):
                    self.visible_volume[j] = np.abs(self.volume_faulty_syndromes[j] -self.volume_faulty_syndromes[j-1])

            # reset the completed actions
            self.completed_actions = np.zeros(self.num_actions,int)
            self.completed_action_count = 0

        else:
            # Update the completed actions, but not the visible syndrome
            self.completed_actions[action] = float(not(self.completed_actions[action]))
        
        for j in range(self.volume_depth):
            self.state[:,:,j] = self.padding_syndrome(self.visible_volume[j])
        for k in range(self.n_action_layers):
            self.state[:,:,self.volume_depth + k] = self.padding_actions(self.completed_actions[k*self.d**2:(k+1)*self.d**2] )

        if self.baselines:
            return self.state, reward, done, new_error_flag
        else:
            return self.state, reward, done, {"new_error_flag": new_error_flag}



    def reset(self):
        self.current_hidden_state = np.zeros((self.d,self.d),int)
        self.volume_hidden_states = [np.zeros((self.d,self.d),int) for j in range(self.volume_depth)] 
        self.volume_true_syndromes = [np.zeros((self.d+1,self.d+1),int) for j in range(self.volume_depth)]
        self.volume_faulty_syndromes = [np.zeros((self.d+1,self.d+1),int) for j in range(self.volume_depth)]
        self.visible_volume = [np.zeros((self.d+1,self.d+1),int) for j in range(self.volume_depth)]                          # This may be increments, or faulty syndromes

        self.completed_actions = np.zeros(self.num_actions,int)             
        self.completed_action_count = 0  

        self.state = np.zeros((self.observation_space.shape),int)
        self.base_syndrome = np.zeros((self.d+1,self.d+1),int)
        for j in range(self.volume_depth):
            self.state[:,:,j] = self.padding_syndrome(self.base_syndrome)
        return self.state

class Surface_Code_Environment_Single_Decoding_Cycle():
    """
    This environment has:

        - a syndrome volume + completed actions as the state - in a form suitable for convnets
        - a variable number of moves. In particular, the game ends if the agent does the identity:
            - If at this stage the agent is in the ground state, then the game is won - reward = +1
            - If at tihs stage the agent is not in the ground state, then the game is lost - reward = -1
        - NB: There is NO static decoder in this environment!
        - NB: We don't even allow the option of using baselines - this is for keras-rl only!
        - NBNB: This environment has channels first!!!
        - can deal with error_model in {"X","DP"}
        - can deal with faulty syndromes - i.e. p_meas > 0

    TODO: flesh out documentation!

    Parameters
    ----------


    Returns
    -------

    """


    def __init__(self, d=5, p_phys=0.01, p_meas=0.01, error_model="DP", use_Y=True, volume_depth=3, training=True, static_decoder=None):

        self.d = d
        self.p_phys = p_phys
        self.p_meas= p_meas
        self.error_model = error_model
        self.use_Y = use_Y
        self.volume_depth = volume_depth
        self.training=training
        self.static_decoder = static_decoder

        self.n_action_layers = 0
        if error_model == "X":
            self.num_actions = d**2 + 1
            self.n_action_layers = 1
        elif error_model == "DP":
            if use_Y:
                self.num_actions = 3*d**2 + 1
                self.n_action_layers = 3
            else:
                self.num_actions = 2*d**2 + 1 
                self.n_action_layers = 2
        else:
            print("specified error model not currently supported!")

        self.identity_index = self.num_actions -1
        self.identity_indicator = self.generate_identity_indicator(self.d)

        self.qubits = generateSurfaceCodeLattice(self.d)
        self.qubit_stabilizers = self.get_stabilizer_list(self.qubits, self.d)  
        self.qubit_neighbours = self.get_qubit_neighbour_list(self.d) 
        self.completed_actions = np.zeros(self.num_actions, int)
        
    
        self.observation_space=gym.spaces.Box(low=0,high=1,
                                      shape=(self.volume_depth+self.n_action_layers,
                                        2*self.d+1, 
                                        2*self.d+1),
                                      dtype=np.uint8)
        
        self.action_space = gym.spaces.Discrete(self.num_actions)

        self.hidden_state = np.zeros((self.d, self.d), int)
        self.summed_syndrome_volume = None         
        self.board_state = np.zeros((self.volume_depth + self.n_action_layers, 2 * self.d + 1, 2 * self.d + 1),int)

        self.completed_actions = np.zeros(self.num_actions, int)
        self.acted_on_qubits = set()
        self.legal_actions = set()
        self.done = False

    def reset(self):
        """
        In this environment reset introduces a non-trivial error
        """

        self.completed_actions = np.zeros(self.num_actions, int)
        self.acted_on_qubits = set()
        self.legal_actions = set()
        self.done = False
        
        
        if self.training:
            non_trivial_error = False
            while not non_trivial_error:
                # Let the opponent do it's initial evil
                non_trivial_error = self.initialize_state()
        else:
            non_trivial_error = self.initialize_state()


        # Update the legal moves available to us - at the moment it is only ones that are next to violated stabilizers

        if not non_trivial_error:
            self.legal_actions.add(self.identity_index)
        else:
            legal_qubits = set()

            for qubit_number in range(self.d**2):

                # first we deal with qubits that are adjacent to violated stabilizers
                if self.is_adjacent_to_syndrome(qubit_number):
                    legal_qubits.add(qubit_number)

            # now we have to make a list out of it and account for different types of actions
            self.legal_actions.add(self.identity_index)
            for j in range(self.n_action_layers):
                for legal_qubit in legal_qubits:
                    self.legal_actions.add(legal_qubit + j*self.d**2)

        return self.board_state

        
    def step(self, action):

        done_identity = False
        if action == self.identity_index or int(self.completed_actions[action]) == 1:
            done_identity = True

        # perform the action
        action_lattice = index_to_move(self.d, action, self.error_model, self.use_Y)
        self.hidden_state = obtain_new_error_configuration(self.hidden_state, action_lattice)

        # Update the completed actions and legal moves
        self.completed_actions[action] = int(not(self.completed_actions[action]))
        if not action == self.identity_index:

            acted_qubit = action%(self.d**2)
            
            if acted_qubit not in self.acted_on_qubits:
                self.acted_on_qubits.add(acted_qubit)
                for neighbour in self.qubit_neighbours[acted_qubit]:
                        for j in range(self.n_action_layers):
                            self.legal_actions.add(neighbour + j*self.d**2)

            
        # update the board state to reflect the action thats been taken
        for k in range(self.n_action_layers):
                self.board_state[self.volume_depth + k, :, :] = self.padding_actions(self.completed_actions[k * self.d ** 2:(k + 1) * self.d ** 2])

        if done_identity:
            self.board_state = self.indicate_identity(self.board_state)

        # check what the reward should be
        reward = 0 
        self.done = False

    
        if done_identity:
            # If you have just done the identity, then we check whether you have won or lost

            current_true_syndrome = generate_surface_code_syndrome_NoFT_efficient(self.hidden_state, self.qubits)
            num_anyons = np.sum(current_true_syndrome)
            correct_label = generate_one_hot_labels_surface_code(self.hidden_state, self.error_model)


            if np.argmax(correct_label) == 0 and num_anyons == 0:
                # if you are back in the ground state you win
                self.done = True
                reward = 1
            else:
                # if you are not back in the ground state you lose
                self.done = True
                reward = -1

        elif self.static_decoder is not None:

            # If you didn't do the identity, then we still check that you can be decoded
            current_true_syndrome = generate_surface_code_syndrome_NoFT_efficient(self.hidden_state, self.qubits)
            num_anyons = np.sum(current_true_syndrome)
            correct_label = generate_one_hot_labels_surface_code(self.hidden_state, self.error_model)

            current_true_syndrome_vector = np.reshape(current_true_syndrome,(self.d+1)**2) 
            decoder_label = self.static_decoder.predict(np.array([current_true_syndrome_vector]), batch_size=1, verbose=0)

            if np.argmax(decoder_label[0]) != np.argmax(correct_label):
                self.done = True
                reward = -1


        return self.board_state, reward, self.done, {}

    def is_adjacent_to_syndrome(self, qubit_number):

        for stabilizer in self.qubit_stabilizers[qubit_number]:
            if self.summed_syndrome_volume[stabilizer] != 0:
                return True

        return False

    def initialize_state(self):
        '''
            Creates an initial volume of errors

        '''

        self.hidden_state = np.zeros((self.d, self.d), int)
        self.summed_syndrome_volume = np.zeros((self.d + 1, self.d + 1), int) 
        self.board_state = np.zeros((self.volume_depth + self.n_action_layers, 2 * self.d + 1, 2 * self.d + 1),int)
        
        non_trivial_error = True

        for j in range(self.volume_depth):
            error = generate_error(self.d, self.p_phys, self.error_model)
            self.hidden_state = obtain_new_error_configuration(self.hidden_state, error)
            current_true_syndrome = generate_surface_code_syndrome_NoFT_efficient(self.hidden_state, self.qubits)
            current_faulty_syndrome = generate_faulty_syndrome(current_true_syndrome, self.p_meas)
            self.summed_syndrome_volume += current_faulty_syndrome
            padded_faulty_syndrome = self.padding_syndrome(current_faulty_syndrome)

            # update the board state to reflect the measured syndromes
            self.board_state[j, :, :] = padded_faulty_syndrome

        if int(np.sum(self.summed_syndrome_volume)) == 0:
            non_trivial_error = False

        return non_trivial_error


    def padding_syndrome(self, syndrome_in):
        syndrome_out = np.zeros((2*self.d+1, 2*self.d+1),int)
        
        for x in range( 2*self.d+1 ):
            for y in range( 2*self.d+1 ):

                #label the boundaries and corners
                if x==0 or x== 2*self.d:
                    if y%2 == 1:
                        syndrome_out[x,y] = 1

                if y==0 or y== 2*self.d:
                    if x%2 == 1:
                        syndrome_out[x,y] = 1

                if x%2 == 0 and y%2 == 0: 
                    # copy in the syndrome
                    syndrome_out[ x, y ] = syndrome_in[ int(x/2), int(y/2) ]
                elif x%2 == 1 and y%2 == 1:
                    if (x+y)%4 == 0:
                        #label the stabilizers
                        syndrome_out[x,y] = 1
        return syndrome_out
        
    def padding_actions(self,actions_in):
        # works only for a subset of actions (only x actions)
        # has to be called 1, 2 or 3 separate times, depending on noise-model and use_Y; for X, Z and Y actions
        actions_out = np.zeros( ( 2*self.d+1, 2*self.d+1 ),int )

        for action_index, action_taken in enumerate( actions_in ):
            if action_taken:
                row = int( action_index / self.d )
                col = int( action_index % self.d )

                actions_out[ int( 2*row+1 ), int( 2*col+1 ) ] = 1

        return actions_out

    def indicate_identity(self, board_state):

        for k in range(self.n_action_layers):
            board_state[self.volume_depth + k,:, :] = board_state[self.volume_depth + k,:, :] + self.identity_indicator

        return board_state

    def get_qubit_stabilizer_list(self, qubits, qubit):
        """"
        Given a qubit specification [qubit_row, qubit_column], this function returns the list of non-trivial stabilizer locations adjacent to that qubit
        """

        qubit_stabilizers = []
        row = qubit[0]
        column = qubit[1]
        for j in range(4):
            if qubits[row,column,j,:][2] != 0:     # i.e. if there is a non-trivial stabilizer at that site
                qubit_stabilizers.append(tuple(qubits[row,column,j,:][:2]))
        return qubit_stabilizers  

    def get_stabilizer_list(self, qubits, d):
        """"
        Given a lattice, this function outputs a list of non-trivial stabilizers adjacent to each qubit in the lattice, indexed row-wise starting from top left
        """
        stabilizer_list = []
        for qubit_row in range(self.d):
            for qubit_column in range(self.d):
                stabilizer_list.append(self.get_qubit_stabilizer_list(qubits,[qubit_row,qubit_column]))
        return stabilizer_list

    def get_qubit_neighbour_list(self, d):

        count = 0
        qubit_dict = {}
        qubit_neighbours = []
        for row in range(d):
            for col in range(d):
                qubit_dict[str(tuple([row,col]))] = count
                cells = starmap(lambda a,b: (row+a, col+b), product((0,-1,+1), (0,-1,+1)))
                qubit_neighbours.append(list(cells)[1:])
                count +=1
            
        neighbour_list = []
        for qubit in range(d**2):
            neighbours = []
            for neighbour in qubit_neighbours[qubit]:
                if str(neighbour) in qubit_dict.keys():
                    neighbours.append(qubit_dict[str(neighbour)])
            neighbour_list.append(neighbours)

        return neighbour_list

    def generate_identity_indicator(self, d):

        identity_indicator = np.ones((2 *d + 1, 2 *d + 1),int)
        for j in range(d):
            row = 2*j + 1
            for k in range(d):
                col = 2*k + 1
                identity_indicator[row,col] = 0
        return identity_indicator

# ------------ (8) ----------------------------------------------------------


class Surface_Code_Environment_Multi_Decoding_Cycles():
    """
    This environment has:

        - a syndrome volume + completed actions as the state 
        - a variable number of moves:
            - an error is introduced if the agent does the identity
            - an error is introduced if the agent repeats the same move twice
        - a plus 1 reward for every action that results in the code being in the ground state
        - can deal with error_model in {"X","DP"}
        - can deal with faulty syndromes - i.e. p_meas > 0

    TODO: flesh out documentation!

    Parameters
    ----------


    Returns
    -------

    """


    def __init__(self, d=5, p_phys=0.01, p_meas=0.01, error_model="DP", use_Y=True, volume_depth=3, static_decoder=None):

        self.d = d
        self.p_phys = p_phys
        self.p_meas= p_meas
        self.error_model = error_model
        self.use_Y = use_Y
        self.volume_depth = volume_depth
        self.static_decoder = static_decoder

        self.n_action_layers = 0
        if error_model == "X":
            self.num_actions = d**2 + 1
            self.n_action_layers = 1
        elif error_model == "DP":
            if use_Y:
                self.num_actions = 3*d**2 + 1
                self.n_action_layers = 3
            else:
                self.num_actions = 2*d**2 + 1 
                self.n_action_layers = 2
        else:
            print("specified error model not currently supported!")

        self.identity_index = self.num_actions -1
        self.identity_indicator = self.generate_identity_indicator(self.d)

        self.qubits = generateSurfaceCodeLattice(self.d)
        self.qubit_stabilizers = self.get_stabilizer_list(self.qubits, self.d)  
        self.qubit_neighbours = self.get_qubit_neighbour_list(self.d) 
        self.completed_actions = np.zeros(self.num_actions, int)
        
    
        self.observation_space=gym.spaces.Box(low=0,high=1,
                                      shape=(self.volume_depth+self.n_action_layers,
                                        2*self.d+1, 
                                        2*self.d+1),
                                      dtype=np.uint8)
        
        self.action_space = gym.spaces.Discrete(self.num_actions)

        self.hidden_state = np.zeros((self.d, self.d), int)
        self.current_true_syndrome = np.zeros((self.d+1, self.d+1), int)
        self.summed_syndrome_volume = None         
        self.board_state = np.zeros((self.volume_depth + self.n_action_layers, 2 * self.d + 1, 2 * self.d + 1),int)

        self.completed_actions = np.zeros(self.num_actions, int)
        self.acted_on_qubits = set()
        self.legal_actions = set()
        self.done = False
        self.lifetime = 0

        self.multi_cycle = True

    def reset(self):
        """
        In this environment reset introduces a non-trivial error
        """

        self.done = False
        self.lifetime = 0
        
        # Create the initial error - we wait until there is a non-trivial syndrome. BUT, the lifetime is still updated!
        self.initialize_state()

        # Update the legal moves available to us
        self.reset_legal_moves()

        return self.board_state


    def step(self, action):

        new_error_flag = False
        done_identity = False
        if action == self.identity_index or int(self.completed_actions[action]) == 1:
            done_identity = True

        # 1) Apply the action to the hidden state
        action_lattice = index_to_move(self.d, action, self.error_model, self.use_Y)
        self.hidden_state = obtain_new_error_configuration(self.hidden_state, action_lattice)

        # 2) Calculate the reward
        self.current_true_syndrome = generate_surface_code_syndrome_NoFT_efficient(self.hidden_state, self.qubits)
        current_true_syndrome_vector = np.reshape(self.current_true_syndrome,(self.d+1)**2) 
        num_anyons = np.sum(self.current_true_syndrome)

        correct_label = generate_one_hot_labels_surface_code(self.hidden_state, self.error_model)
        decoder_label = self.static_decoder.predict(np.array([current_true_syndrome_vector]), batch_size=1, verbose=0)

        reward = 0

        if np.argmax(correct_label) == 0 and num_anyons == 0:
            reward = 1.0
        elif np.argmax(decoder_label[0]) != np.argmax(correct_label):
            self.done = True


        # 3) If necessary, apply multiple errors and obtain an error volume
        if done_identity:

            trivial_volume = True
            while trivial_volume: 
                self.summed_syndrome_volume = np.zeros((self.d + 1, self.d + 1), int)
                faulty_syndromes = []
                for j in range(self.volume_depth):
                    error = generate_error(self.d, self.p_phys, self.error_model)
                    if int(np.sum(error)!=0):
                        self.hidden_state = obtain_new_error_configuration(self.hidden_state, error)
                        self.current_true_syndrome = generate_surface_code_syndrome_NoFT_efficient(self.hidden_state, self.qubits)
                    current_faulty_syndrome = generate_faulty_syndrome(self.current_true_syndrome, self.p_meas)
                    faulty_syndromes.append(current_faulty_syndrome)
                    self.summed_syndrome_volume += current_faulty_syndrome
                    self.lifetime += 1

                if int(np.sum(self.summed_syndrome_volume)) != 0:
                    trivial_volume = False

            for j in range(self.volume_depth):
                self.board_state[j, :, :] = self.padding_syndrome(faulty_syndromes[j])


            # reset the completed actions
            self.reset_legal_moves()

            # update the part of the state which shows the actions you have just taken
            self.board_state[self.volume_depth:,:,:] = np.zeros((self.n_action_layers, 2 * self.d + 1, 2 * self.d + 1),int)


        else:
            # Update the completed actions and legal moves
            self.completed_actions[action] = int(not(self.completed_actions[action]))
            if not action == self.identity_index:

                acted_qubit = action%(self.d**2)
                
                if acted_qubit not in self.acted_on_qubits:
                    self.acted_on_qubits.add(acted_qubit)
                    for neighbour in self.qubit_neighbours[acted_qubit]:
                            for j in range(self.n_action_layers):
                                self.legal_actions.add(neighbour + j*self.d**2)

                
            # update the board state to reflect the action thats been taken
            for k in range(self.n_action_layers):
                    self.board_state[self.volume_depth + k, :, :] = self.padding_actions(self.completed_actions[k * self.d ** 2:(k + 1) * self.d ** 2])


        return self.board_state, reward, self.done, {}

    def initialize_state(self):
        '''
            Creates an initial volume of errors

        '''

        self.done = False
        self.hidden_state = np.zeros((self.d, self.d), int)
        self.current_true_syndrome = np.zeros((self.d+1, self.d+1), int) 
        self.board_state = np.zeros((self.volume_depth + self.n_action_layers, 2 * self.d + 1, 2 * self.d + 1),int)
        
        trivial_volume = True
        while trivial_volume:
            self.summed_syndrome_volume = np.zeros((self.d + 1, self.d + 1), int)
            faulty_syndromes = []
            for j in range(self.volume_depth):
                error = generate_error(self.d, self.p_phys, self.error_model)
                if int(np.sum(error)) != 0:
                    self.hidden_state = obtain_new_error_configuration(self.hidden_state, error)
                    self.current_true_syndrome = generate_surface_code_syndrome_NoFT_efficient(self.hidden_state, self.qubits)
                current_faulty_syndrome = generate_faulty_syndrome(self.current_true_syndrome, self.p_meas)
                faulty_syndromes.append(current_faulty_syndrome)
                self.summed_syndrome_volume += current_faulty_syndrome
                self.lifetime += 1

            if int(np.sum(self.summed_syndrome_volume)) != 0:
                trivial_volume = False

        # update the board state to reflect the measured syndromes
        for j in range(self.volume_depth):
            self.board_state[j, :, :] = self.padding_syndrome(faulty_syndromes[j])


    def reset_legal_moves(self):

        self.completed_actions = np.zeros(self.num_actions, int)
        self.acted_on_qubits = set()
        self.legal_actions = set()

        legal_qubits = set()
        for qubit_number in range(self.d**2):

            # first we deal with qubits that are adjacent to violated stabilizers
            if self.is_adjacent_to_syndrome(qubit_number):
                legal_qubits.add(qubit_number)

        # now we have to make a list out of it and account for different types of actions
        self.legal_actions.add(self.identity_index)
        for j in range(self.n_action_layers):
            for legal_qubit in legal_qubits:
                self.legal_actions.add(legal_qubit + j*self.d**2)



    def is_adjacent_to_syndrome(self, qubit_number):

        for stabilizer in self.qubit_stabilizers[qubit_number]:
            if self.summed_syndrome_volume[stabilizer] != 0:
                return True

        return False

    def padding_syndrome(self, syndrome_in):
        syndrome_out = np.zeros((2*self.d+1, 2*self.d+1),int)
        
        for x in range( 2*self.d+1 ):
            for y in range( 2*self.d+1 ):

                #label the boundaries and corners
                if x==0 or x== 2*self.d:
                    if y%2 == 1:
                        syndrome_out[x,y] = 1

                if y==0 or y== 2*self.d:
                    if x%2 == 1:
                        syndrome_out[x,y] = 1

                if x%2 == 0 and y%2 == 0: 
                    # copy in the syndrome
                    syndrome_out[ x, y ] = syndrome_in[ int(x/2), int(y/2) ]
                elif x%2 == 1 and y%2 == 1:
                    if (x+y)%4 == 0:
                        #label the stabilizers
                        syndrome_out[x,y] = 1
        return syndrome_out
        
    def padding_actions(self,actions_in):
        # works only for a subset of actions (only x actions)
        # has to be called 1, 2 or 3 separate times, depending on noise-model and use_Y; for X, Z and Y actions
        actions_out = np.zeros( ( 2*self.d+1, 2*self.d+1 ),int )

        for action_index, action_taken in enumerate( actions_in ):
            if action_taken:
                row = int( action_index / self.d )
                col = int( action_index % self.d )

                actions_out[ int( 2*row+1 ), int( 2*col+1 ) ] = 1

        return actions_out

    def indicate_identity(self, board_state):

        for k in range(self.n_action_layers):
            board_state[self.volume_depth + k,:, :] = board_state[self.volume_depth + k,:, :] + self.identity_indicator

        return board_state

    def get_qubit_stabilizer_list(self, qubits, qubit):
        """"
        Given a qubit specification [qubit_row, qubit_column], this function returns the list of non-trivial stabilizer locations adjacent to that qubit
        """

        qubit_stabilizers = []
        row = qubit[0]
        column = qubit[1]
        for j in range(4):
            if qubits[row,column,j,:][2] != 0:     # i.e. if there is a non-trivial stabilizer at that site
                qubit_stabilizers.append(tuple(qubits[row,column,j,:][:2]))
        return qubit_stabilizers  

    def get_stabilizer_list(self, qubits, d):
        """"
        Given a lattice, this function outputs a list of non-trivial stabilizers adjacent to each qubit in the lattice, indexed row-wise starting from top left
        """
        stabilizer_list = []
        for qubit_row in range(self.d):
            for qubit_column in range(self.d):
                stabilizer_list.append(self.get_qubit_stabilizer_list(qubits,[qubit_row,qubit_column]))
        return stabilizer_list

    def get_qubit_neighbour_list(self, d):

        count = 0
        qubit_dict = {}
        qubit_neighbours = []
        for row in range(d):
            for col in range(d):
                qubit_dict[str(tuple([row,col]))] = count
                cells = starmap(lambda a,b: (row+a, col+b), product((0,-1,+1), (0,-1,+1)))
                qubit_neighbours.append(list(cells)[1:])
                count +=1
            
        neighbour_list = []
        for qubit in range(d**2):
            neighbours = []
            for neighbour in qubit_neighbours[qubit]:
                if str(neighbour) in qubit_dict.keys():
                    neighbours.append(qubit_dict[str(neighbour)])
            neighbour_list.append(neighbours)

        return neighbour_list

    def generate_identity_indicator(self, d):

        identity_indicator = np.ones((2 *d + 1, 2 *d + 1),int)
        for j in range(d):
            row = 2*j + 1
            for k in range(d):
                col = 2*k + 1
                identity_indicator[row,col] = 0
        return identity_indicator





