#----- (0) Imports ---------------------------------------------------------------------------------------------------------------

from Function_Library import *
import gym
import copy
from itertools import product, starmap

#---------- (1) --------------------------------------------------------------------------------------------------------------------------------------

class Surface_Code_Environment_Multi_Decoding_Cycles():
    """
    A surface code environment for obtaining decoding agents in the fault-tolerant setting.
    In particular:

        - The visible state consists of a syndrome volume + completed action volume
        - The agent can perform single qubit Pauli flips on physical data qubits
        -   - an error volume is introduced if the agent does the identity
            - an error volume is introduced if the agent repeats the same move twice
        - a plus 1 reward is given for every action that results in the code being in the ground state space.
        - The terminal state criterion is satisfied for every hidden state that cannot be decoded by the referee/static decoder

    Note, this environment can:
        - cater for error_model in {"X","DP"}
        - cater for faulty syndromes - i.e. p_meas > 0

    Also, this environment provides all methods as required by an openAi gym class. In particular:
        - reset
        - step


    Attributes
    ----------

    :param: d: The code distance
    :param: p_phys: The physical error probability on a single physical data qubit
    :param: p_meas: The measurement error probability on a single syndrome bit
    :param: error_model: A string in ["X, DP"]
    :param: use_Y: A boolean indicating whether the environment accepts Y Pauli flips as actions
    :param: volume_depth: The number of sequential syndrome measurements performed when generating a new syndrome volume.
    :param: static_decoder: A homology class predicting decoder for perfect syndromes.

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
        Resetting of the environment introduces a new non-trivial syndrome volume.

        :return: self.board_state: The new reset visible state of the environment = syndrome volume + blank action history volume
        """

        self.done = False
        self.lifetime = 0
        
        # Create the initial error - we wait until there is a non-trivial syndrome. BUT, the lifetime is still updated!
        self.initialize_state()

        # Update the legal moves available to us
        self.reset_legal_moves()

        return self.board_state


    def step(self, action):
        """
        Given an action, this method executes the logic of the environment.

        :param: action: The action, given as an integer, supplied by some agent.
        :return: self.board_state: The new reset visible state of the environment = syndrome volume + action history volume
        :return: reward: The reward for the action
        :return: self.done: The boolean terminal state indicator
        :return: info: A dictionary via which additional diagnostic information can be provided. Empty here.
        """

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


        # 3) If necessary, apply multiple errors and obtain an error volume - ensure that a non-trivial volume is generated
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


            # reset the legal moves
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
        """
        Generate an initial non-trivial syndrome volume
        """

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
        """
        Reset the legal moves
        """

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
        """
        Determine whether a qubit is adjacent to a violated stabilizer
        """

        for stabilizer in self.qubit_stabilizers[qubit_number]:
            if self.summed_syndrome_volume[stabilizer] != 0:
                return True

        return False

    def padding_syndrome(self, syndrome_in):
        """
        Pad a syndrome into the required embedding
        """

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
        """
        Pad an action history for a single type of Pauli flip into the required embedding.
        """
        actions_out = np.zeros( ( 2*self.d+1, 2*self.d+1 ),int )

        for action_index, action_taken in enumerate( actions_in ):
            if action_taken:
                row = int( action_index / self.d )
                col = int( action_index % self.d )

                actions_out[ int( 2*row+1 ), int( 2*col+1 ) ] = 1

        return actions_out

    def indicate_identity(self, board_state):
        """
        Pad the action history to indicate that an identity has been performed.
        """

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
        """"
        Given a lattice, this function provides a list of the neighbouring qubits for each physical qubit.
        """

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
        """"
        A simple helper function to generate the array that will be added to the action history to indicate that an identity has been performed.
        """

        identity_indicator = np.ones((2 *d + 1, 2 *d + 1),int)
        for j in range(d):
            row = 2*j + 1
            for k in range(d):
                col = 2*k + 1
                identity_indicator[row,col] = 0
        return identity_indicator





