# ------------ Helper Functions -------------------------------------------------------------------------
#
# This file provides all necessary helper functions. 
#
# ----- (0) Imports --------------------------------------------------------------------------------------

import random
import numpy as np

# ---- (1) Functions -------------------------------------------------------------------------------------


def generateSurfaceCodeLattice(d):
    """"
    This function generates a distance d square surface code lattice. in particular, the function returns 
    an array which, for each physical qubit, details the code-stabilizers supported on that qubit. To be more
    precise:
    
     - qubits[i,j,:,:] is a 4x3 array describing all the stabilizers supported on physical qubit(i,j)
           -> for the surface code geometry each qubit can support up to 4 stabilizers
     - qubits[i,j,k,:] is a 3-vector describing the k'th stabilizer supported on physical qubit(i,j)
           -> qubits[i,j,k,:] = [x_lattice_address, y_lattice_address, I or X or Y or Z]
    
    :param: d: The lattice width and height (or, equivalently, for the surface code, the code distance)
    :return: qubits: np.array listing and describing the code-stabilizers supported on each qubit
    """
    
    if np.mod(d,2) != 1:
        raise Exception("for the surface code d must be odd!")
        
    qubits = [ [ [
                   [ x, y, ((x+y)%2)*2+1],
                   [ x, y+1, ((x+y+1)%2)*2+1],
                   [ x+1, y, ((x+1+y)%2)*2+1],
                   [ x+1, y+1, ((x+1+y+1)%2)*2+1]
                ] for y in range(d)] for x in range(d)]
    qubits = np.array(qubits)
    
    for x in range(d):
        for y in range(d):
            for k in range(4):
                if (qubits[x,y,k,0] == 0 and qubits[x,y,k,1]%2 == 0):
                    qubits[x,y,k,2] = 0
                if (qubits[x,y,k,0] == d and qubits[x,y,k,1]%2 == 1):
                    qubits[x,y,k,2] = 0
                    
                if (qubits[x,y,k,1] == 0 and qubits[x,y,k,0]%2 == 1):
                    qubits[x,y,k,2] = 0
                if (qubits[x,y,k,1] == d and qubits[x,y,k,0]%2 == 0):
                    qubits[x,y,k,2] = 0
    return qubits


def multiplyPaulis(a,b):
    """"
    A simple helper function for multiplying Pauli Matrices. Returns ab.
    :param: a: an int in [0,1,2,3] representing [I,X,Y,Z]
    :param: b: an int in [0,1,2,3] representing [I,X,Y,Z]
    """
    
    out = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
    return out[int(a)][int(b)]


# 2) Error generation

def generate_error(d,p_phys,error_model):
    """"
    This function generates an error configuration, via a single application of the specified error channel, on a square dxd lattice.
    
    :param: d: The code distance/lattice width and height (for surface/toric codes)
    :param: p_phys: The physical error rate.
    :param: error_model: A string in ["X", "DP", "IIDXZ"] indicating the desired error model.
    :return: error: The error configuration
    """
    
    if error_model == "X":
        return generate_X_error(d,p_phys)
    elif error_model == "DP":
        return generate_DP_error(d,p_phys)
    elif error_model == "IIDXZ":
        return generate_IIDXZ_error(d,p_phys)
        
    return error

def generate_DP_error(d,p_phys):
    """"
    This function generates an error configuration, via a single application of the depolarizing noise channel, on a square dxd lattice.
    
    :param: d: The code distance/lattice width and height (for surface/toric codes)
    :param: p_phys: The physical error rate.
    :return: error: The error configuration
    """

    error = np.zeros((d,d),int) 
    for i in range(d): 
        for j in range(d):
            p = 0
            if np.random.rand() < p_phys:
                p = np.random.randint(1,4)
                error[i,j] = p
                
    return error

def generate_X_error(d,p_phys):
    """"
    This function generates an error configuration, via a single application of the bitflip noise channel, on a square dxd lattice.
    
    :param: d: The code distance/lattice width and height (for surface/toric codes)
    :param: p_phys: The physical error rate.
    :return: error: The error configuration
    """
    
    
    error = np.zeros((d,d),int) 
    for i in range(d): 
        for j in range(d):
            p = 0
            if np.random.rand() < p_phys:
                error[i,j] = 1
    
    return error
                
def generate_IIDXZ_error(d,p_phys):
    """"
    This function generates an error configuration, via a single application of the IIDXZ noise channel, on a square dxd lattice.
    
    :param: d: The code distance/lattice width and height (for surface/toric codes)
    :param: p_phys: The physical error rate.
    :return: error: The error configuration
    """
    
    error = np.zeros((d,d),int)
    for i in range(d):
        for j in range(d):
            X_err = False
            Z_err = False
            p = 0
            if np.random.rand() < p_phys:
                X_err = True
                p = 1
            if np.random.rand() < p_phys:
                Z_err = True
                p = 3
            if X_err and Z_err:
                p = 2

            error[i,j] = p
    
    return error

def generate_surface_code_syndrome_NoFT_efficient(error,qubits):
    """"
    This function generates the syndrome (violated stabilizers) corresponding to the input error configuration, 
    for the surface code.
    
    :param: error: An error configuration on a square lattice
    :param: qubits: The qubit configuration
    :return: syndrome: The syndrome corresponding to input error
    """
    
    d = np.shape(error)[0]
    syndrome = np.zeros((d+1,d+1),int)

    for i in range(d): 
        for j in range(d):
            if error[i,j] != 0:
                for k in range(qubits.shape[2]):
                    if qubits[i,j,k,2] != error[i,j] and qubits[i,j,k,2] != 0:
                        a = qubits[i,j,k,0]
                        b = qubits[i,j,k,1]
                        syndrome[a,b] = 1 - syndrome[a,b]
                        
    return syndrome

def generate_faulty_syndrome(true_syndrome, p_measurement_error):
    """"
    This function takes in a true syndrome, and generates a faulty syndrome according to some
    given probability of measurement errors.
    
    :param: true_syndrome: The original perfect measurement syndrome
    :return: p_measurement_error: The probability of measurement error per stabilizer
    :return: faulty_syndrome: The faulty syndrome
    """
    
    faulty_syndrome = np.zeros(np.shape(true_syndrome),int)

    # First we take care of the "bulk stabilizers"
    for row in range(1, true_syndrome.shape[0]-1):
        for col in range(1,true_syndrome.shape[1]-1):
            if np.random.rand() < p_measurement_error:
                faulty_syndrome[row,col] = 1 - true_syndrome[row,col]
            else:
                faulty_syndrome[row,col] = true_syndrome[row,col]

    # Now we take care of the boundary stabilizers
    row = 0
    for col in [2*x +1 for x in range(int(true_syndrome.shape[0]/2 - 1))]:
        if np.random.rand() < p_measurement_error:
                faulty_syndrome[row,col] = 1 - true_syndrome[row,col]
        else:
            faulty_syndrome[row,col] = true_syndrome[row,col]
    row = true_syndrome.shape[0] - 1
    for col in [2*x + 2 for x in range(int(true_syndrome.shape[0]/2 - 1))]:
        if np.random.rand() < p_measurement_error:
                faulty_syndrome[row,col] = 1 - true_syndrome[row,col]
        else:
            faulty_syndrome[row,col] = true_syndrome[row,col]

    col = 0
    for row in [2*x + 2 for x in range(int(true_syndrome.shape[0]/2 - 1))]:
        if np.random.rand() < p_measurement_error:
                faulty_syndrome[row,col] = 1 - true_syndrome[row,col]
        else:
            faulty_syndrome[row,col] = true_syndrome[row,col]
    col = true_syndrome.shape[0] - 1
    for row in [2*x +1 for x in range(int(true_syndrome.shape[0]/2 - 1))]:
        if np.random.rand() < p_measurement_error:
                faulty_syndrome[row,col] = 1 - true_syndrome[row,col]
        else:
            faulty_syndrome[row,col] = true_syndrome[row,col]
  
    return faulty_syndrome


def obtain_new_error_configuration(old_configuration,new_gates):
    """"
    This function generates a new error configuration out of an old configuration and a new configuration,
     which might arise either from errors or corrections.
    
    :param: old_configuration: An error configuration on a square lattice
    :param: new_gates: An error configuration on a square lattice
    :return: new_configuration: The resulting error configuration
    """
    
    new_configuration = np.zeros(np.shape(old_configuration))
    for row in range(new_configuration.shape[0]):
        for col in range(new_configuration.shape[1]):
            new_configuration[row,col] = multiplyPaulis(new_gates[row,col], old_configuration[row,col])
            
    return new_configuration

def index_to_move(d,move_index,error_model,use_Y=True):
    """"
    Given an integer index corresponding to a Pauli flip on a physical data qubit, this
    function generates the lattice representation of the move.
    
    :param: d: The code distance
    :param: move_index: The integer representation of the Pauli Flip
    :param: error_model: A string in ["X", "DP", "IIDXZ"] indicating the desired error model.
    :param: use_Y: a boolean indicating whether or not Y flips are allowed
    :return: new_move: A lattice representation of the desired move.
    """

    new_move = np.zeros((d,d))
    
    if error_model == "X":
        if move_index < (d**2):

            move_type = 1
            new_index = move_index
            row = new_index/d
            col = new_index%d

            new_move[int(row),int(col)] = move_type

    elif error_model == "DP":
        if use_Y:
            if move_index < (d**2)*3:
                move_type = int(move_index/d**2) + 1
                new_index = move_index - (move_type - 1)*d**2

                row = new_index/d
                col = new_index%d

                new_move[int(row),int(col)] = move_type
        else:
            if move_index < (d**2)*2:
                move_type = int(move_index/d**2) + 1

                new_index = move_index - (move_type - 1)*d**2

                row = new_index/d
                col = new_index%d

                if move_type == 2:
                    move_type = 3

                new_move[int(row),int(col)] = move_type

    else:
        print("Error model you have specified is not currently supported")

    return new_move

def generate_one_hot_labels_surface_code(error,err_model):
    """"

    This function generates the homology class label, in a one-hot encoding, for a given perfect syndrome, to use as the target label
    for a feed forward neural network homology class predicting decoder.

    :param: error: An error configuration on a square lattice
    :param: err_model: A string in ["IIDXZ","DP","X"]
    :return: training_label: The one-encoded training label
    """
    
    d = error.shape[0]
    
    X = 0
    Z = 0
        
    for x in range(d):
        if error[x,0] == 1 or error[x,0] == 2:
            X = 1 - X
    for y in range(d):
        if error[0,y] == 3 or error[0,y] == 2:
            Z = 1 - Z
            
    if err_model in ["IIDXZ","DP"]:
        training_label = np.zeros(4,int)                       
    else:
        training_label = np.zeros(2,int)

    training_label[X + 2*Z] = 1
    
    return training_label
