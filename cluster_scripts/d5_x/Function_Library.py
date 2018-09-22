# This script contains stable, concise, well documented helper functions!
# Best to not edit anything in here lightly!

import random
import numpy as np
from numpy import copy
import scipy
import pickle

import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential, Model
from keras.regularizers import l1_l2, l2
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Lambda, Cropping2D, Activation, LSTM, Input, merge, Concatenate
from keras import backend as K
from keras.optimizers import Adam

import PIL as pil
from PIL import Image as im
from PIL import ImageDraw as imdraw
from PIL import ImageFont


def generateToricCodeLattice(d):
    """"
    This function generates a distance d toric code lattice. in particular, the function returns 
    an array which, for each physical qubit, details the code-stabilizers supported on that qubit. To be more
    precise:
    
     - qubits[i,j,:,:] is a 4x3 array describing all the stabilizers supported on physical qubit(i,j)
           -> for the toric code geometry each qubit can support up to 4 stabilizers
     - qubits[i,j,k,:] is a 3-vector describing the k'th stabilizer supported on physical qubit(i,j)
           -> qubits[i,j,k,:] = [x_lattice_address, y_lattice_address, I or X or Y or Z]
    
    TO DO: 1) make a simple diagram illlustrating the surface code lattice indexing conventions.
    
    :param: d: The lattice width and height (or, equivalently, for the surface code, the code distance)
    :return: qubits: np.array listing and describing the code-stabilizers supported on each qubit
    """
    
    if np.mod(d,2) != 0:
        raise Exception("for the toric code d must be even!")
    
    Lx = d
    Ly = d
    
    qubits = [ [ [
                   [ (i-1)%Lx, (j-1)%Ly, (((i-1)%Lx+(j-1)%Ly)%2)*2+1],
                   [ (i-1)%Lx, j, (((i-1)%Lx+j)%2)*2+1],
                   [ i, (j-1)%Ly, ((i+(j-1)%Ly)%2)*2+1],
                   [ i, j, ((i+j)%2)*2+1]
                ] for j in range(Ly)] for i in range(Lx)]

    return np.array(qubits) 

def generateSurfaceCodeLattice(d):
    """"
    This function generates a distance d square surface code lattice. in particular, the function returns 
    an array which, for each physical qubit, details the code-stabilizers supported on that qubit. To be more
    precise:
    
     - qubits[i,j,:,:] is a 4x3 array describing all the stabilizers supported on physical qubit(i,j)
           -> for the surface code geometry each qubit can support up to 4 stabilizers
     - qubits[i,j,k,:] is a 3-vector describing the k'th stabilizer supported on physical qubit(i,j)
           -> qubits[i,j,k,:] = [x_lattice_address, y_lattice_address, I or X or Y or Z]
    
    TO DO: 1) make a simple diagram illlustrating the surface code lattice indexing conventions.
    
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
    """
    
    out = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
    return out[int(a)][int(b)]


# 2) Error generation

def generate_error(d,p_phys,error_model):
    """"
    This function generates an error configuration, from a single time step, due to bit flip noise, on a 
    square lattice.
    
    :param: d: The code distance/lattice width and height (for surface/toric codes)
    :param: p_phys: The physical error rate.
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
    This function generates an error configuration, from a single time step, due to depolarizing noise,
    on a square lattice.
    
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
    This function generates an error configuration, from a single time step, due to bit flip noise, on a 
    square lattice.
    
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
    This function generates an error configuration, from a single time step, due to bit IID bit flip and
    dephasing noise, on a square lattice.
    
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

def generate_sequential_error(p_phys, previous_error, err_model):
    """"
    This function generates the next error in an error sequence, given the most recent error.
    
    :param: p_phys: The physical error rate.
    :param: previous_error: The most recent error in the sequence
    :param: err_model: The error model -i.e. a string in ["IIDXZ","DP","X"]
    :return: new_error: The new/current error configuration
    """
    
    d = previous_error.shape[0]
    
    if err_model == "IIDXZ":
        raw_new_error = generate_IIDXZ_error(d,p_phys)
    elif err_model == "DP":
        raw_new_error = generate_DP_error(d,p_phys)
    elif err_model == "X":
        raw_new_error = generate_X_error(d,p_phys)
        
    new_error = np.zeros(np.shape(previous_error))
    for row in range(new_error.shape[0]):
        for col in range(new_error.shape[1]):
            new_error[row,col] = multiplyPaulis(raw_new_error[row,col], previous_error[row,col])
            
    return new_error


# 3) Syndrome Generation

def generate_surface_code_syndrome_NoFT(error):
    """"
    This function generates the syndrome (violated stabilizers) corresponding to the input error, 
    for the surface code.
    
    :param: error: An error configuration on a square lattice
    :return: syndrome: The syndrome corresponding to input error
    """
    
    d = np.shape(error)[0]
    syndrome = np.zeros((d+1,d+1),int)
    
    qubits = generateSurfaceCodeLattice(d)

    for i in range(d): 
        for j in range(d):
            if error[i,j] != 0:
                for k in range(qubits.shape[2]):
                    if qubits[i,j,k,2] != error[i,j] and qubits[i,j,k,2] != 0:
                        a = qubits[i,j,k,0]
                        b = qubits[i,j,k,1]
                        syndrome[a,b] = 1 - syndrome[a,b]
                        
    return syndrome

def generate_toric_code_syndrome_NoFT(error):
    """"
    This function generates the syndrome (violated stabilizers) corresponding to the input error, 
    for the toric code.
    
    :param: error: An error configuration on a square lattice
    :return: syndrome: The syndrome corresponding to input error
    """
    
    d = np.shape(error)[0]
    syndrome = np.zeros((d,d),int)
    
    qubits = generateToricCodeLattice(d)

    for i in range(d): 
        for j in range(d):
            if error[i,j] != 0:
                for k in range(qubits.shape[2]):
                    if qubits[i,j,k,2] != error[i,j]:
                        a = qubits[i,j,k,0]
                        b = qubits[i,j,k,1]
                        syndrome[a,b] = 1 - syndrome[a,b]
                        
    return syndrome


# 4) Label generation

def generate_one_hot_labels_surface_code(error,err_model):
    """"
    This function generates the NN target label, in a one-hot encoding (suitable for a categorical cross
    entropy loss function) for the given error and error model, assuming a surface code toplogy (i.e. one
    logicl qubit).
    
    TO DO: Add some more detail about what exactly these target variables are!
    
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
    
def generate_one_hot_labels_toric_code(error,err_model):
    """"
    This function generates the NN target label, in a one-hot encoding (suitable for a categorical cross
    entropy loss function) for the given error and error model, assuming a toric code topology (i.e. two 
    logical qubits).
    
    TO DO: Add some more detail about what exactly these target variables are!
    
    :param: error: An error configuration on a square lattice
    :param: err_model: A string in ["IIDXZ","DP","X"]
    :return: training_label: The one-hot encoded training label
    """
    
    d = error.shape[0]
    
    X1 = 0
    X2 = 0
    Z1 = 0
    Z2 = 0

    for i in range(d):
        if error[i,0] == 3 or error[i,0] == 2:
            Z1 = 1 - Z1
        if error[i,0] == 1 or error[i,0] == 2:
            X2 = 1 - X2
    for j in range(d):
        if error[0,j] == 3 or error[0,j] == 2:
            Z2 = 1 - Z2
        if error[0,j] == 1 or error[0,j] == 2:
            X1 = 1 - X1

    if err_model in ["DP","IIDXZ"]:
        training_label = np.zeros(16,int)                       
    else:
        training_label = np.zeros(4,int)

    training_label[X1 + 2*X2 + 4*Z1 + 8*Z2] = 1   
    
    return training_label
    
def generate_direct_parity_labels_surface_code(error, err_model):
    """"
    TO DO: Add a good description of this target label
    
    :param: error: An error configuration on a square lattice
    :param: err_model: A string in ["IIDXZ","DP","X"]
    :return: training_label: The training label
    """
    
    d = error.shape[0]
    
    if err_model in ["IIDXZ","DP"]:
        training_label = np.ones(2*d,int)                      
    elif err_model == "X":
        training_label = np.ones(d,int)
        
    for x in range(d):
        for y in range(d):
            if error[x,y] == 1 or error[x,y] == 2:
                training_label[y] *= -1
            if (error[x,y] == 3 or error[x,y] == 2) and err_model != "X":
                training_label[x+d] *= -1
    
    return training_label

def generate_direct_parity_labels_toric_code(error, err_model):
    """"
    TO DO: Add a good description of this target label
    TO DO: Implement this function :)
    
    :param: error: An error configuration on a square lattice
    :param: err_model: A string in ["IIDXZ","DP","X"]
    :return: training_label: The training label
    """
    
    raise Exception("Function not yet implemented")    
    
    return None


# 5) Training pair and dataset generation

def generateTrainingPairs_NoFT(d,code_type,err_model,p_phys,label_type):
    """"
    This function generates a single:
             - error
             - syndrome
             - training_label
    
    for a particular configuration of code type, error model and label type.
    
    :param: d: The code distance/lattice width and height (for surface/toric codes)
    :param: code_type: A string in ["surface","toric"]
    :param: err_model: A string in ["DP", "IIDXZ", "X"]
    :param: p_phys: The physical error rate, as interpreted by the specified error model
    :param: label_type: A string in ["one_hot","direct_parity"]
    :return: error: A specification of the error that occured
    :return: syndrome: A list of the violated stabilizers
    :return: training_label: The training_label/target for the NN 
    """
    
    # 0) Verify that parameters have been validly specified
    
    if err_model not in ["DP", "IIDXZ", "X"]:
        raise Exception("An invalid error model has been specified.")
        
    if code_type not in ["surface","toric"]:
        raise Exception("An invalid code type has been specified.")
        
    if label_type not in ["one_hot","direct_parity"]:
        raise Exception("An invalid label type has been specified")
        
    
    # 1) Generate the error configuration:
    
    if err_model == "DP":
        error = generate_DP_error(d,p_phys)
    elif err_model == "IIDXZ":
        error = generate_IIDXZ_error(d,p_phys)
    elif err_model == "X":
        error = generate_X_error(d,p_phys)
        
    # 2) Generate the syndrome
    
    if code_type == "surface":
        syndrome = generate_surface_code_syndrome_NoFT(error)
    elif code_type == "toric":
        syndrome = generate_toric_code_syndrome_NoFT(error)
        
    # 3) Generate the training label
    
    if code_type == "surface":
        if label_type == "one_hot":
            training_label = generate_one_hot_labels_surface_code(error,err_model)
        elif label_type == "direct_parity":
            training_label = generate_direct_parity_labels_surface_code(error, err_model)
    elif code_type == "toric":
        if label_type == "one_hot":
            training_label = generate_one_hot_labels_toric_code(error,err_model)
        elif label_type == "direct_parity":
            training_label = generate_direct_parity_labels_toric_code(error, err_model)
    
    # 4) Return the desired outputs
    
    return error, syndrome, training_label

def generate_FT_training_sequence(d,code_type,err_model,p_phys,label_type, p_measurement_error, seq_length):
    """"
    This function generates a sequence (array) of:
             - errors
             - syndromes
             - faulty syndromes
             - training_label
    
    for a particular configuration of code type, error model and label type.
    
    :param: d: The code distance/lattice width and height (for surface/toric codes)
    :param: code_type: A string in ["surface","toric"]
    :param: err_model: A string in ["DP", "IIDXZ", "X"]
    :param: p_phys: The physical error rate, as interpreted by the specified error model
    :param: label_type: A string in ["one_hot","direct_parity"]
    :param: p_measurement_error: The probability of a stabilizer measurement error
    :param: seq_length: The length of the syndrome collection sequence
    :return: error_sequence: The sequence of errors
    :return: true_syndrome_sequence: A sequence of true (perfect) syndromes
    :return: faulty_syndrome_sequence: A sequence of faulty syndromes
    :return: training_label: the training label corresponding to the final syndrome
    """
    
    # 0) Verify that parameters have been validly specified
    
    if err_model not in ["DP", "IIDXZ", "X"]:
        raise Exception("An invalid error model has been specified.")
        
    if code_type not in ["surface","toric"]:
        raise Exception("An invalid code type has been specified.")
        
    if label_type not in ["one_hot","direct_parity"]:
        raise Exception("An invalid label type has been specified")
        
    
    # 1) Generate the error configuration:
    
    # 1a) Generate the first error in the sequence 
    
    if err_model == "DP":
        initial_error = generate_DP_error(d,p_phys)
    elif err_model == "IIDXZ":
        initial_error = generate_IIDXZ_error(d,p_phys)
    elif err_model == "X":
        initial_error = generate_X_error(d,p_phys)
        
    # 1b) Generate and store the subsequent sequence
    
    error_sequence = np.zeros((seq_length,d,d))
    error_sequence[0,:,:] = initial_error
    
    for j in range(1,seq_length):
        error_sequence[j,:,:] = generate_sequential_error(p_phys, error_sequence[j-1,:,:], err_model)
        
    # 2) Generate the true (no measurement error) syndrome sequence
    
    if code_type == "surface":
        true_syndrome_sequence = np.zeros((seq_length,d+1,d+1))
        for j in range(seq_length):
            true_syndrome_sequence[j,:,:] = generate_surface_code_syndrome_NoFT(error_sequence[j,:,:])
    elif code_type == "toric":
        true_syndrome_sequence = np.zeros((seq_length,d,d))
        for j in range(seq_length):
            true_syndrome_sequence[j,:,:] = generate_toric_code_syndrome_NoFT(error_sequence[j,:,:])
            
    # 3) Generate the faulty syndrome sequence
    
    faulty_syndrome_sequence = np.zeros(true_syndrome_sequence.shape)
    
    for j in range(seq_length):
        faulty_syndrome_sequence[j,:,:] = generate_faulty_syndrome(true_syndrome_sequence[j,:,:], p_measurement_error)
        
    # 4) Generate the final training_label
    
    final_error = error_sequence[seq_length -1,:,:]
    if code_type == "surface":
        if label_type == "one_hot":
            training_label = generate_one_hot_labels_surface_code(final_error,err_model)
        elif label_type == "direct_parity":
            training_label = generate_direct_parity_labels_surface_code(final_error, err_model)
    elif code_type == "toric":
        if label_type == "one_hot":
            training_label = generate_one_hot_labels_toric_code(final_error,err_model)
        elif label_type == "direct_parity":
            training_label = generate_direct_parity_labels_toric_code(final_error, err_model)
    
    # 4) Return the desired outputs
    
    return error_sequence, true_syndrome_sequence, faulty_syndrome_sequence, training_label


def generateDataSet_NoFT(n_instances,d,code_type,err_model,p_phys,label_type, flatten=True):
    """"
    This function generates a complete training dataset with labels, suitable for either a convolutional
    or feed forward neural network, condition on the value of "flatten"
    
    :param: n_instances: The number of labelled instances to generate
    :param: d: The code distance/lattice width and height (for surface/toric codes)
    :param: code_type: A string in ["surface","toric"]
    :param: err_model: A string in ["DP", "IIDXZ", "X"]
    :param: p_phys: The physical error rate, as interpreted by the specified error model
    :param: label_type: A string in ["one_hot","direct_parity"]
    :param: flatten: Boolean which determines whether the inputs are given to the NN as a matrix or vector
    :return: instance_set: An array of training instances
    :return: label_set: An array of training_labels corresponding to the training instances   
    """
    
    
    # 1) Determine the size of the _stabilizer_ lattice, which is the input to the NN
    
    if code_type == "surface":
        instance_set = np.zeros((n_instances,d+1,d+1))      
    elif code_type == "toric":
        instance_set = np.zeros((n_instances,d,d))      
    
    # 2) Determine the size of the training labels
  
    if code_type == "toric":
        if label_type == "one_hot":
            if err_model in ["DP","IIDXZ"]:
                label_set = np.zeros((n_instances,16))
            elif err_model == "X":
                label_set = np.zeros((n_instances,4))
        elif label_type == "direct_parity":
            if err_model in ["DP","IIDXZ"]:
                label_set = np.zeros((n_instances,2*d))
            elif err_model == "X":
                label_set = np.zeros((n_instances,d))
    elif code_type == "surface":
        if label_type == "one_hot":
            if err_model in ["DP","IIDXZ"]:
                label_set = np.zeros((n_instances,4))
            elif err_model == "X":
                label_set = np.zeros((n_instances,2))
        elif label_type == "direct_parity":
            if err_model in ["DP","IIDXZ"]:
                label_set = np.zeros((n_instances,2*d))
            elif err_model == "X":
                label_set = np.zeros((n_instances,d))

    # 3) Create the dataset
            
    for j in range(n_instances):
        error, instance_set[j,:,:], label_set[j,:] = generateTrainingPairs_NoFT(d,code_type,err_model,
                                                                                p_phys,label_type)
        
    # 4) Shape the data as required for a convolutional or FF NN.
    
    if flatten:
        if code_type == "surface":
            instance_set = np.reshape(instance_set,(n_instances,(d+1)*(d+1)))
        elif code_type == "toric":
            instance_set = np.reshape(instance_set,(n_instances,(d)*(d)))
    else:
        # add dummy dimension for the keras CNN channel convention
        instance_set = np.expand_dims(instance_set,3)
        
    return instance_set, label_set

def generateDataSet_FT(n_instances,d,code_type,err_model,p_phys,label_type, 
                       p_measurement_error, seq_length, flatten=True):
    """"
    This function generates a complete training dataset with labels, suitable for either a convolutional
    or feed forward neural network, condition on the value of "flatten"
    
    :param: n_instances: The number of labelled instances to generate
    :param: d: The code distance/lattice width and height (for surface/toric codes)
    :param: code_type: A string in ["surface","toric"]
    :param: err_model: A string in ["DP", "IIDXZ", "X"]
    :param: p_phys: The physical error rate, as interpreted by the specified error model
    :param: label_type: A string in ["one_hot","direct_parity"]
    :param: p_measurement_error: The probability of a stabilizer measurement error
    :param: seq_length: The length of the syndrome collection sequence
    :param: flatten: Boolean which determines whether the inputs are given to the NN as a matrix or vector
    :return: instance_set: An array of training instances
    :return: label_set: An array of training_labels corresponding to the training instances   
    """
    
    
    # 1) Determine the size of the _stabilizer_ lattice, which is the input to the NN
    
    if code_type == "surface":
        true_syndrome_set = np.zeros((n_instances,seq_length, d+1,d+1))
        faulty_syndrome_set = np.zeros((n_instances,seq_length, d+1,d+1))  
    elif code_type == "toric":
        true_syndrome_set = np.zeros((n_instances,seq_length, d,d))
        faulty_syndrome_set = np.zeros((n_instances,seq_length, d,d))        
    
    # 2) Determine the size of the training labels
    
    if code_type == "toric":
        if label_type == "one_hot":
            if err_model in ["DP","IIDXZ"]:
                label_set = np.zeros((n_instances,16))
            elif err_model == "X":
                label_set = np.zeros((n_instances,4))
        elif label_type == "direct_parity":
            if err_model in ["DP","IIDXZ"]:
                label_set = np.zeros((n_instances,2*d))
            elif err_model == "X":
                label_set = np.zeros((n_instances,d))
    elif code_type == "surface":
        if label_type == "one_hot":
            if err_model in ["DP","IIDXZ"]:
                label_set = np.zeros((n_instances,4))
            elif err_model == "X":
                label_set = np.zeros((n_instances,2))
        elif label_type == "direct_parity":
            if err_model in ["DP","IIDXZ"]:
                label_set = np.zeros((n_instances,2*d))
            elif err_model == "X":
                label_set = np.zeros((n_instances,d))
            
    # 3) Create the dataset
            
    for j in range(n_instances):
        error_seq, true_syndrome_set[j,:], faulty_syndrome_set[j,:], label_set[j,:] = generate_FT_training_sequence(d,
                                                                                                code_type,err_model,p_phys,label_type, p_measurement_error, seq_length)
        
    # 4) Shape the data as required for a convolutional or FF NN.
    
    if flatten:
        if code_type == "surface":
            true_syndrome_set = np.reshape(true_syndrome_set,(n_instances,seq_length,(d+1)*(d+1)))
            faulty_syndrome_set = np.reshape(faulty_syndrome_set,(n_instances,seq_length,(d+1)*(d+1)))
        elif code_type == "toric":
            true_syndrome_set = np.reshape(true_syndrome_set,(n_instances,seq_length,d**2))
            faulty_syndrome_set = np.reshape(faulty_syndrome_set,(n_instances,seq_length,d**2))
    else:
        # add dummy dimension for the keras CNN channel convention
        true_syndrome_set = np.expand_dims(true_syndrome_set,3)
        faulty_syndrome_set = np.expand_dims(true_syndrome_set,3)
        
    return true_syndrome_set, faulty_syndrome_set, label_set

# 6) Neural network related helper functions

def build_CNN_surface_code(image_width, num_categories,c_layers, ff_layers, padding, label_type, initial_lr):
    """"
    This function builds and compiles a Convolutional Neural Network, which is suitable for decoding
    the surface code.In particular it possible to specify the structure of both the convolutional and 
    feed forward layers, using the following convention:
    
      - c_layers is a list of lists. Each inner list has form [num_kernels, kernel_width, output_dropout]
      - ff_layers is a list of lists. Each inner list has the form [neurons, output_dropout]
    
    :param: image_width: The width of the input image (i.e. stabilizer lattice)
    :param: num_categories: The length of the target label
    :param: c_layers: A specification of the convolutional structure of the network, in the form given above.
    :param: ff_layers: A specification of the feed forward structure of the network, in the form given above.
    :param: padding: String in ["valid","same"] which determines whether or not zero padding is utilized
    :param: label_type: A string in ["one_hot","direct_parity"]
    :param: initial_lr: The initial learning rate
    :return: cnn: The convolutional neural network, as a Keras sequential object  
    """
    
    # 0) Check that the inputs are valid
    
    if label_type not in ["one_hot","direct_parity"]:
        raise Exception("An invalid label type has been specified")
        
    if padding not in ["valid","same"]:
        raise Exception("Padding has been invalidly specified")
        
    # 1) Construct the CNN
    
    cnn = Sequential()

    # a) Add the convolutional layers, with droupout
    for j in range(len(c_layers)):

        # Add the convolutional layers
        if j == 0:
            cnn.add(Conv2D(c_layers[j][0], c_layers[j][1], activation='relu',padding=padding,input_shape=(image_width,image_width,1)))
        else:
            cnn.add(Conv2D(c_layers[j][0], c_layers[j][1], activation='relu',padding=padding))
        
        # Add the dropout (NB: Be careful here)
        if c_layers[j][2] > 0:
            cnn.add(Dropout(c_layers[j][2]))
            
    # 2) Flatten in preparation for the dense layers
    cnn.add(Flatten())
    
    # 3) Add the FF Layers
    for j in range(len(ff_layers)):
        # a) The dense layer 
        cnn.add(Dense(ff_layers[j][0], activation='relu'))
        
        # b) The dropout (conditional on being greater than 0)
        if ff_layers[j][1] > 0:
            cnn.add(Dropout(ff_layers[j][1]))
            
    # 4) Add the final output layer and compile
    
    if initial_lr == "default":
        ad_opt = Adam()
    else:
        ad_opt = Adam(lr=initial_lr)
            
    if label_type == "one_hot":
        cnn.add(Dense(num_categories,activation="softmax"))
        cnn.compile(optimizer=ad_opt,loss='categorical_crossentropy',metrics=['accuracy'])
    elif label_type == "direct_parity":
        cnn.add(Dense(num_categories,activation="tanh"))
        cnn.compile(optimizer=ad_opt,loss='mean_squared_error')
    
    return cnn

def tiled_padding(image_batch):
    """"
    A simple helper function which creates a 2x2 tiling of an image. This tiling can be used
    to construct convolutional neural networks which see images with periodic/toric boundary conditions
    
    :param: image_batch: A batch of images
    :return: tiled_image_batch: A batch of tiled images
    """
    
    wide_image_batch = K.concatenate([image_batch,image_batch],axis=2)
    tiled_image_batch = K.concatenate([wide_image_batch,wide_image_batch],axis=1)
    return tiled_image_batch

def tiled_padding(image_batch):
    """"
    A simple helper function which creates a 2x2 tiling of an image. This tiling can be used
    to construct convolutional neural networks which see images with periodic/toric boundary conditions
    
    :param: image_batch: A batch of images
    :return: tiled_image_batch: A batch of tiled images
    """
    
    wide_image_batch = K.concatenate([image_batch,image_batch],axis=2)
    tiled_image_batch = K.concatenate([wide_image_batch,wide_image_batch],axis=1)
    return tiled_image_batch

def build_CNN_toric_code(image_width, num_categories,c_layers, ff_layers, label_type, initial_lr):
    """"
    This function builds and compiles a Convolutional Neural Network, which is suitable for decoding the
    toric code due to a cute "toric padding" procedure. In particular it possible to specify
    the structure of both the convolutional and feed forward layers, using the following convention:
    
      - c_layers is a list of lists. Each inner list has form [num_kernels, kernel_width, output_dropout]
      - ff_layers is a list of lists. Each inner list has the form [neurons, output_dropout]
    
    :param: image_width: The width of the input image (i.e. stabilizer lattice)
    :param: num_categories: The length of the target label
    :param: c_layers: A specification of the convolutional structure of the network, in the form given above.
    :param: ff_layers: A specification of the feed forward structure of the network, in the form given above.
    :param: label_type: A string in ["one_hot","direct_parity"]
    :param: initial_lr: The initial learning rate
    :return: cnn: The convolutional neural network, as a Keras sequential object  
    """
    
    # 0) Check that the inputs are valid
    
    if label_type not in ["one_hot","direct_parity"]:
        raise Exception("An invalid label type has been specified")
        
    if padding not in ["valid","same"]:
        raise Exception("Padding has been invalidly specified")
        
    # 1) Construct the CNN
    
    cnn = Sequential()

    # 1) Add the convolutional layers
    for j in range(len(c_layers)):
        
        # a) Tile-pad the image
        if j == 0:
            cnn.add(Lambda(tiled_padding,input_shape=(image_width,image_width,1)))
        else:
            cnn.add(Lambda(tiled_padding))
        
        # b) Crop the image so that the convolutional kernel experiences the image as a torus
        crop_width = image_width-c_layers[j][1] + 1
        cnn.add(Cropping2D(((0,crop_width),(0,crop_width))))
        
        # c) Add the convolutional layers
        cnn.add(Conv2D(c_layers[j][0], c_layers[j][1], activation='relu'))
                    
        # d) Add the dropout (NB: Be careful here)
        if c_layers[j][2] > 0:
            cnn.add(Dropout(c_layers[j][2]))
            
    # 2) Flatten in preparation for the dense layers
    cnn.add(Flatten())
    
    # 3) Add the FF Layers
    for j in range(len(ff_layers)):
        # a) The dense layer 
        cnn.add(Dense(ff_layers[j][0], activation='relu'))
        
        # b) The dropout (conditional on being greater than 0)
        if ff_layers[j][1] > 0:
            cnn.add(Dropout(ff_layers[j][1]))
    
    # 4) Add the final output layer and compile
    
    if initial_lr == "default":
        ad_opt = Adam()
    else:
        ad_opt = Adam(lr=initial_lr)
            
    if label_type == "one_hot":
        cnn.add(Dense(num_categories,activation="softmax"))
        cnn.compile(optimizer=ad_opt,loss='categorical_crossentropy',metrics=['accuracy'])
    elif label_type == "direct_parity":
        cnn.add(Dense(num_categories,activation="tanh"))
        cnn.compile(optimizer=ad_opt,loss='mean_squared_error')
    
    return cnn
def build_FFNN(num_features, num_categories,ff_layers,label_type, initial_lr):
    """"
    This function builds and compiles a Feed Forward Neural Network. In particular it possible to specify
    the structure of the feed forward layers arbitrarily, using the following convention:
    
      - ff_layers is a list of lists. Each inner list has the form [neurons, output_dropout]
    
    :param: num_features: the number of input features to the NN
    :param: num_categories: The length of the target label
    :param: ff_layers: A specification of the feed forward structure of the network, in the form given above.
    :param: label_type: A string in ["one_hot","direct_parity"]
    :return: nnet: The feed forward neural network, as a Keras sequential object  
    """
    
    # 0) Check that the inputs are valid
    
    if label_type not in ["one_hot","direct_parity"]:
        raise Exception("An invalid label type has been specified")
        
    # 1) Build the NN
    
    nnet = Sequential()
    
    # 1a) Bulk Layers
    for layer in range(len(ff_layers)):
        
        if layer == 0:
            nnet.add(Dense(ff_layers[0][0], activation='relu', input_shape=(num_features,)))
        else:
            nnet.add(Dense(ff_layers[layer][0], activation='relu'))
                
        if ff_layers[layer][1] > 0:
            nnet.add(Dropout(ff_layers[layer][1]))
                     
    # 1b) Final Output Layer
    
    if initial_lr == "default":
        ad_opt = Adam()
    else:
        ad_opt = Adam(lr=initial_lr)
                     
    if label_type == "one_hot":
        nnet.add(Dense(num_categories,activation="softmax"))
        nnet.compile(optimizer=ad_opt,loss='categorical_crossentropy',metrics=['accuracy'])
    elif label_type == "direct_parity":
        nnet.add(Dense(num_categories,activation="tanh"))
        nnet.compile(optimizer=ad_opt,loss='mean_squared_error')
    
    return nnet

def train_NN(nn, dataset, labelset, epochs, val_ratio, stopping_patience, batch_size, label_type):
    """"
    A basic helper function for training a neural network. Basically a wrapper to Keras Sequential object
    fit method, but with an additional early stopping callback to determine convergence.

    :param: nn: A keras sequential object
    :param: dataset: Array of training instances
    :param: labelset: Arrau of training labels
    :param: epochs: Maximum number of allowed training epochs
    :param: val_ratio: The percentage of training instances to be used as a validation set
    :param: stopping_patience: Training will stop after this number of epochs if not improvement has been achieved
    :param: batch_size: The training batch size
    :param: label_type: a string in ["one_hot", "direct_parity"]
    :return: nn: The trained keras sequential object
    """

    to_monitor = "val_acc"
    if label_type == "direct_parity":
        to_monitor = "val_loss"

    early_stopping = EarlyStopping(monitor=to_monitor, mode='auto', patience=stopping_patience)

    nn.fit(dataset, labelset,
           batch_size=batch_size,
           epochs=epochs,
           verbose=1,
           validation_split=val_ratio,
           callbacks=[early_stopping])

    return nn

def train_NN_with_LR_decay(nn, dataset,labelset, epochs, val_ratio, stopping_patience, batch_size,
                           lr_decay_factor,lr_patience, lr_cooldown, lr_min, label_type):
    """"
    A basic helper function for training a neural network. Basically a wrapper to Keras Sequential object
    fit method, but with an additional early stopping callback to determine convergence.
    
    :param: nn: A keras sequential object
    :param: dataset: Array of training instances
    :param: labelset: Arrau of training labels
    :param: epochs: Maximum number of allowed training epochs
    :param: val_ratio: The percentage of training instances to be used as a validation set
    :param: stopping_patience: Training will stop after this number of epochs if not improvement has been achieved
    :param: batch_size: The training batch size
    :param: lr_decay_factor: The factor by which learning rate is decayed after no improvement in lr_patience number of epochs
    :param: lr_patience: The learning rate will be decayed if no improvement in relevant metric is seen after this number of epochs
    :param: lr_cooldown: The number of epochs after a decay during which improvement will not be monitored
    :param: lr_min: The minimum learning rate allowed
    :param: label_type: a string in ["one_hot", "direct_parity"]
    :return: nn: The trained keras sequential object 
    """
    
    to_monitor = "val_acc"
    if label_type == "direct_parity":
        to_monitor = "val_loss"
        
    early_stopping = EarlyStopping(monitor=to_monitor, mode='auto',patience=stopping_patience)

    LR_schedule = ReduceLROnPlateau(monitor=to_monitor, 
                                      factor=lr_decay_factor, 
                                      patience=lr_patience, 
                                      verbose=1, 
                                      mode='auto', 
                                      epsilon=0.0001, 
                                      cooldown=lr_cooldown, min_lr=lr_min)

    nn.fit(dataset, labelset,
           batch_size=batch_size,
           epochs=epochs,
           verbose=1,
           validation_split=0.1,
           callbacks = [early_stopping, LR_schedule])
    
    return nn


def evaluate_NN_one_hot(nn, testset, testlabels):
    """"
    A basic helper function for evaluating a trained neural network whose outputs are categorical probabilities
    """
    
    return nn.evaluate(testset,testlabels)[1]

def evaluate_NN_direct_parity(nn, testset, testlabels):
    """"
    A helper function for evaluating a trained neural network whose outputs are direct parity predictions.
    """
    
    eval_preds = nn.predict(testset)
    
    return evaluate_prediction_accuracy_direct_parity(eval_preds,testlabels)

def correct_indicator(prediction, target):
    """"
    A helper function for determining whether a direct parity prediction resulted in a correct decoding.
    """

    d = len(target)/2
    
    x_pred = prediction[:d]
    z_pred = prediction[d:]
    
    x_t = target[:d]
    z_t = target[d:]
    
    x_max = np.max(np.abs(x_pred))
    x_ind = list(np.abs(x_pred)).index(x_max)
    
    z_max = np.max(np.abs(z_pred))
    z_ind = list(np.abs(z_pred)).index(z_max)
    
    is_correct = False
    
    if np.sign(x_pred[x_ind]) == np.sign(x_t[x_ind]) and np.sign(z_pred[z_ind]) == np.sign(z_t[z_ind]):
        is_correct = True
    
    return is_correct

def evaluate_prediction_accuracy_direct_parity(pred_array, target_array):
    """"
    A helper function for evaluating the percentage of direct parity predictions in an array of predictions
    which would have led to correct decodings.
    """
    
    num_instances = pred_array.shape[0]
    indicator_list = [correct_indicator(pred_array[j,:],target_array[j,:]) for j in range(num_instances)]
    num_correct = np.sum(indicator_list)
    final_acc = num_correct/(num_instances*1.0)
    return final_acc


# ---------- RL specific functions -----------------------

def generate_surface_code_syndrome_NoFT_efficient(error,qubits):
    """"
    This function generates the syndrome (violated stabilizers) corresponding to the input error, 
    for the surface code.
    
    :param: error: An error configuration on a square lattice
    :param: qubits: The qubit configuration - this function efficient because this is not generated everycall.
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

def obtain_new_error_configuration(old_configuration,new_gates):
    """"
    This function generates a new error configurarion out of an old configuration and a new set of
    gates which might arise either from errors or corrections.
    
    :param: old_configuration: An error configuration on a square lattice
    :param: new_gates: A configuration of new gates, either from an error or active corrections
    :return: new_configuration: The resulting error configuration
    """
    
    new_configuration = np.zeros(np.shape(old_configuration))
    for row in range(new_configuration.shape[0]):
        for col in range(new_configuration.shape[1]):
            new_configuration[row,col] = multiplyPaulis(new_gates[row,col], old_configuration[row,col])
            
    return new_configuration

def X_surface_code_environment_delayed_syndrome(qubits, hidden_state, action, 
                                                time_step, delta_t_p_phys, p_phys, FFNN):
    """"
    This function performs the role of the env with which the RL learner interacts. More specifically,
    given the current "hidden_state" - i.e. the error configuration on the lattice - and the action 
    which is specified by the RL learner, the environment then generates a new hidden state, a new visible
    state (anyon syndrome) and a reward. In particular, the environment will also introduce errors at
    set time intervals, according to the specified physical error rate.
    
    NBNB: To do this properly, this should really be a class and all necessary dynamic properties can be 
    attributes which then do not have to be passed around.
    
    :param: qubits: The qubit configuration
    :param: hidden_state: The current error configuration
    :param: action: The correction gates specified by the agent
    :param: time_step: The current time step in the episode
    :param: delta_t_p_phys: The time interval in steps between physical error processes
    :param: p_phys: The physical error rate
    :param: T_thresh: The anyon threshold for terminal states, as a percentage
    :return: new_hidden_state: The new error configuration
    :return: new_visible_state: The new syndrome
    :return: reward: The new reward
    :return: error: The error that occured, if any
    :return: terminal_state: A boolean which indicates whether or not it is game over
    """
    
    d = hidden_state.shape[0]
    new_syndrome = False
    
    # Apply error/ do syndrome measurement
    if time_step%delta_t_p_phys == 0:
        error = generate_X_error(d,p_phys)
        hidden_state = obtain_new_error_configuration(hidden_state,error)
        new_syndrome = True
        
    # Apply the action, and obtain the new configuration
    new_hidden_state = obtain_new_error_configuration(hidden_state,action)
    new_visible_state = generate_surface_code_syndrome_NoFT_efficient(new_hidden_state,qubits)
    new_visible_state_vec = np.reshape(new_visible_state,(d+1)**2)
    num_anyons = np.sum(new_visible_state)
    
    correct_label = generate_one_hot_labels_surface_code(new_hidden_state,"X")
    FFNN_label = FFNN.predict(np.array([new_visible_state_vec]), batch_size=1, verbose=0)
    terminal_state = False
    
    reward = 0
    
    if np.argmax(correct_label) == 0 and num_anyons == 0:
        # If you have not made a logical operation and there are no anyons then get a reward
        reward = 1
    elif np.argmax(FFNN_label[0]) != np.argmax(correct_label):
        # If the new state cannot be decoded then the episode is over
        terminal_state = True

    return new_hidden_state, new_visible_state, reward, terminal_state, new_syndrome


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

class Memory:   # stored as ( s, a, r, s_ , T) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample) 

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond  = K.abs(error) < clip_delta

    squared_loss = 0.5 * K.square(error)
    linear_loss  = clip_delta * (K.abs(error) - 0.5 * clip_delta)

    return tf.where(cond, squared_loss, linear_loss)

def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
    return K.mean(huber_loss(y_true, y_pred, clip_delta))


def build_FF_DQN_with_streams(network_architecture, num_features, num_outputs, learning_rate):
    
    # extract network geometry
    ff_layers_on_syndrome = network_architecture["ff_s"]
    ff_layers_on_actions = network_architecture["ff_a"]
    final_ff_layers = network_architecture["ff_f"]
    
    num_actions = num_outputs
    num_layers_on_syndrome = len(ff_layers_on_syndrome)
    num_layers_on_actions = len(ff_layers_on_actions)
    num_layers_final = len(final_ff_layers)
    
    # create lists to store the layers
    fefo_layers_on_syndrome = []
    fefo_d_layers_on_syndrome = []
    
    fefo_layers_on_actions = []
    fefo_d_layers_on_actions = []
    
    fefo_layers_final = []
    fefo_d_layers_final = []
    
    # 1) specify the inputs:
    syndrome_inputs = Input(shape=(num_features,),name="s_inputs")
    action_inputs = Input(shape=(num_actions,),name="a_inputs")
    
    # 2) if layers are required on the syndrome, add those now:
    if num_layers_on_syndrome > 0:
        for j in range(num_layers_on_syndrome):
            if j == 0:
                fefo_layers_on_syndrome.append(Dense(units=ff_layers_on_syndrome[j][0],
                                                     activation=ff_layers_on_syndrome[j][1],
                                                     kernel_regularizer=l2(ff_layers_on_syndrome[j][2]),
                                                     name="ff_s_"+str(j))(syndrome_inputs))
            else:
                fefo_layers_on_syndrome.append(Dense(units=ff_layers_on_syndrome[j][0],
                                                     activation=ff_layers_on_syndrome[j][1],
                                                     kernel_regularizer=l2(ff_layers_on_syndrome[j][2]),
                                                     name="ff_s_"+str(j))(fefo_d_layers_on_syndrome[j-1]))


            fefo_d_layers_on_syndrome.append(Dropout(ff_layers_on_syndrome[j][3],name="ff_s_d_"+str(j))
                                             (fefo_layers_on_syndrome[j]))
    else:
        # This is a trick that makes it easier to merge the inputs if no pre-processing stream is required
        fefo_d_layers_on_syndrome.append(syndrome_inputs)
        num_layers_on_syndrome = 1
            
    # 3) if layers are required on the actions, add those now:
    if num_layers_on_actions > 0:
        for j in range(num_layers_on_actions):
            if j == 0:
                fefo_layers_on_actions.append(Dense(units=ff_layers_on_actions[j][0],
                                                     activation=ff_layers_on_actions[j][1],
                                                     kernel_regularizer=l2(ff_layers_on_actions[j][2]),
                                                     name="ff_a_"+str(j))(action_inputs))
            else:
                fefo_layers_on_actions.append(Dense(units=ff_layers_on_actions[j][0],
                                                     activation=ff_layers_on_actions[j][1],
                                                     kernel_regularizer=l2(ff_layers_on_actions[j][2]),
                                                     name="ff_a_"+str(j))(fefo_d_layers_on_actions[j-1]))


            fefo_d_layers_on_actions.append(Dropout(ff_layers_on_actions[j][3],name="ff_a_d_"+str(j))
                                            (fefo_layers_on_actions[j]))
    else:
        fefo_d_layers_on_actions.append(action_inputs)
        num_layers_on_actions = 1
        
        
    # 4) Now we concatenate the outputs of the pre-processing streams:
    
    merged_activations = Concatenate(name="merged_activations")([fefo_d_layers_on_syndrome[num_layers_on_syndrome - 1],
                                                                  fefo_d_layers_on_actions[num_layers_on_actions - 1]])
    
    # 5) Finally we can build the merging network
    
    if num_layers_final > 0:
        
        for j in range(num_layers_final):
            if j == 0:
                fefo_layers_final.append(Dense(units=final_ff_layers[j][0],
                                           activation=final_ff_layers[j][1],
                                           kernel_regularizer=l2(final_ff_layers[j][2]),
                                           name="ff_f_"+str(j))(merged_activations))
            else:
                fefo_layers_final.append(Dense(units=final_ff_layers[j][0],
                                            activation=final_ff_layers[j][1],
                                            kernel_regularizer=l2(final_ff_layers[j][2]),
                                            name="ff_f_"+str(j))(fefo_d_layers_final[j-1]))

            fefo_d_layers_final.append(Dropout(final_ff_layers[j][3],name="ff_f_d_"+str(j))(fefo_layers_final[j]))
     
        outputs = Dense(units=num_outputs, 
                        activation="linear",
                        name="predictions")(fefo_d_layers_final[len(fefo_d_layers_final) - 1])
        
        
    else:
        outputs = Dense(units=num_outputs, activation="linear",name="predictions")(merged_activations)
    

    ffnn = Model(inputs=[syndrome_inputs,action_inputs], outputs=outputs)
    
    ad_opt = Adam(lr=learning_rate)
    ffnn.compile(optimizer=ad_opt, loss=huber_loss_mean)
    
    return ffnn

def build_bayesian_FF_DQN_with_streams(network_architecture, num_features, num_outputs, learning_rate):
    
    # This is a network that will use dropout also when predicting!
    #           - note the "training=True" flag when calling Dropout layers
    
    # extract network geometry
    ff_layers_on_syndrome = network_architecture["ff_s"]
    ff_layers_on_actions = network_architecture["ff_a"]
    final_ff_layers = network_architecture["ff_f"]
    
    num_actions = num_outputs
    num_layers_on_syndrome = len(ff_layers_on_syndrome)
    num_layers_on_actions = len(ff_layers_on_actions)
    num_layers_final = len(final_ff_layers)
    
    # create lists to store the layers
    fefo_layers_on_syndrome = []
    fefo_d_layers_on_syndrome = []
    
    fefo_layers_on_actions = []
    fefo_d_layers_on_actions = []
    
    fefo_layers_final = []
    fefo_d_layers_final = []
    
    # 1) specify the inputs:
    syndrome_inputs = Input(shape=(num_features,),name="s_inputs")
    action_inputs = Input(shape=(num_actions,),name="a_inputs")
    
    # 2) if layers are required on the syndrome, add those now:
    if num_layers_on_syndrome > 0:
        for j in range(num_layers_on_syndrome):
            if j == 0:
                fefo_layers_on_syndrome.append(Dense(units=ff_layers_on_syndrome[j][0],
                                                     activation=ff_layers_on_syndrome[j][1],
                                                     kernel_regularizer=l2(ff_layers_on_syndrome[j][2]),
                                                     name="ff_s_"+str(j))(syndrome_inputs))
            else:
                fefo_layers_on_syndrome.append(Dense(units=ff_layers_on_syndrome[j][0],
                                                     activation=ff_layers_on_syndrome[j][1],
                                                     kernel_regularizer=l2(ff_layers_on_syndrome[j][2]),
                                                     name="ff_s_"+str(j))(fefo_d_layers_on_syndrome[j-1]))


            fefo_d_layers_on_syndrome.append(Dropout(ff_layers_on_syndrome[j][3],name="ff_s_d_"+str(j))
                                             (fefo_layers_on_syndrome[j],training=True))
    else:
        # This is a trick that makes it easier to merge the inputs if no pre-processing stream is required
        fefo_d_layers_on_syndrome.append(syndrome_inputs)
        num_layers_on_syndrome = 1
            
    # 3) if layers are required on the actions, add those now:
    if num_layers_on_actions > 0:
        for j in range(num_layers_on_actions):
            if j == 0:
                fefo_layers_on_actions.append(Dense(units=ff_layers_on_actions[j][0],
                                                     activation=ff_layers_on_actions[j][1],
                                                     kernel_regularizer=l2(ff_layers_on_actions[j][2]),
                                                     name="ff_a_"+str(j))(action_inputs))
            else:
                fefo_layers_on_actions.append(Dense(units=ff_layers_on_actions[j][0],
                                                     activation=ff_layers_on_actions[j][1],
                                                     kernel_regularizer=l2(ff_layers_on_actions[j][2]),
                                                     name="ff_a_"+str(j))(fefo_d_layers_on_actions[j-1]))


            fefo_d_layers_on_actions.append(Dropout(ff_layers_on_actions[j][3],name="ff_a_d_"+str(j))
                                            (fefo_layers_on_actions[j],training=True))
    else:
        fefo_d_layers_on_actions.append(action_inputs)
        num_layers_on_actions = 1
        
        
    # 4) Now we concatenate the outputs of the pre-processing streams:
    
    merged_activations = Concatenate(name="merged_activations")([fefo_d_layers_on_syndrome[num_layers_on_syndrome - 1],
                                                                  fefo_d_layers_on_actions[num_layers_on_actions - 1]])
    
    # 5) Finally we can build the merging network
    
    if num_layers_final > 0:
        
        for j in range(num_layers_final):
            if j == 0:
                fefo_layers_final.append(Dense(units=final_ff_layers[j][0],
                                           activation=final_ff_layers[j][1],
                                           kernel_regularizer=l2(final_ff_layers[j][2]),
                                           name="ff_f_"+str(j))(merged_activations))
            else:
                fefo_layers_final.append(Dense(units=final_ff_layers[j][0],
                                            activation=final_ff_layers[j][1],
                                            kernel_regularizer=l2(final_ff_layers[j][2]),
                                            name="ff_f_"+str(j))(fefo_d_layers_final[j-1]))


            fefo_d_layers_final.append(Dropout(final_ff_layers[j][3],
                                               name="ff_f_d_"+str(j))(fefo_layers_final[j],training=True))
        
        outputs = Dense(units=num_outputs, 
                        activation="linear",
                        name="predictions")(fefo_d_layers_final[len(fefo_d_layers_final) - 1])
        
        
    else:
        outputs = Dense(units=num_outputs, activation="linear",name="predictions")(merged_activations)
    

    ffnn = Model(inputs=[syndrome_inputs,action_inputs], outputs=outputs)
    
    ad_opt = Adam(lr=learning_rate)
    ffnn.compile(optimizer=ad_opt, loss=huber_loss_mean)
    
    return ffnn

def index_to_move(d,move_index,error_model,use_Y=True):
    
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

# ------------ episilon-greedy sampling ---------------

def choose_random_move(d,num_actions, identity_percent,error_model,use_Y=True):
    
    if np.random.rand() < identity_percent:
        random_move_index = num_actions-1
    else:
        random_move_index = random.randrange(num_actions)
        
    random_move = index_to_move(d,random_move_index,error_model,use_Y)
    return random_move, random_move_index

def epsilon_greedy_action(Q_network,state,d,epsilon,num_actions, identity_percent,error_model,use_Y=True):
    
    if np.random.rand() < epsilon:
        action, action_index = choose_random_move(d,num_actions, identity_percent,error_model)
    else:
        Q_net_preds = Q_network.predict(state)
        action_index = np.argmax(Q_net_preds)
        action = index_to_move(d,action_index,error_model,use_Y)
    
    return action, action_index

# ------------- boltzmann sampling -----------------------

def build_boltzmann_model(num_actions):
    inputs = Input(shape=(num_actions,))
    outputs = Activation('softmax')(inputs)
    b_model = Model(inputs=inputs, outputs=outputs)
    return b_model
    
def get_boltzmann_distribution(boltzmann_model, Q_net, temp, current_state):
    action_values = Q_net.predict(current_state)
    scaled_action_values = action_values/(temp*1.0)
    b_dist = boltzmann_model.predict(scaled_action_values)[0]
    return b_dist

def boltzmann_action(boltzmann_model, Q_net, temp, current_state,num_actions,d,error_model, use_Y=True):
    b_dist = get_boltzmann_distribution(boltzmann_model, Q_net, temp, current_state)
    b_dist = b_dist/b_dist.sum()             # floating point normalization
    action_index = np.random.choice(num_actions,1,p=b_dist)
    action = index_to_move(d,action_index,error_model,use_Y)
    return action, action_index

# ------------ bayesian sampling -----------------------------

# This is extremely brute force, rebuilding the model everytime, but it seems like the only way that works.

def update_network_architecture(network_architecture,sampling_dropout):
    
    for key in network_architecture.keys():
        if len(network_architecture[key])> 0:
            for layer in network_architecture[key]:
                layer[3] = sampling_dropout
                
    return network_architecture
    

def bayesian_action(active_Q_net, network_architecture, sampling_dropout, 
                    current_state, d, error_model,num_features, num_outputs, learning_rate,use_Y=True):
    
    # First we update the network architecture:
    network_architecture = update_network_architecture(network_architecture,sampling_dropout)
    
    # Build a new "bayesian" network
    bayesian_Q_net = build_bayesian_FF_DQN_with_streams(network_architecture, num_features, num_outputs, learning_rate)
    
    # Set the weights of the new bayesian network to that of the active network
    bayesian_Q_net.set_weights(active_Q_net.get_weights())
    
    # Now we can choose an action using the "bayesian" net
    b_Q_net_preds = bayesian_Q_net.predict(current_state)
    action_index = np.argmax(b_Q_net_preds)
    action = index_to_move(d,action_index,error_model,use_Y)
    
    return action, action_index

# -----------------------------------------------------------
    

def construct_targets_from_batch_double_Q(batch, target_network, active_network, gamma,d,num_actions):
    # batch is a list of tuples (idx, sample_experience)
    
    y_batch, errors = [], []
    batch_size = len(batch)
    
    x_batch_syndrome = np.zeros((batch_size,(d+1)**2))
    x_batch_actions = np.zeros((batch_size,num_actions))
    
    counter = 0
    for idx, sample in batch:
        
        batched_state = sample[0]
        action = sample[1]
        reward = sample[2]
        new_batched_state = sample[3]
        done = sample [4]
        
        target = active_network.predict(batched_state)[0]
        action_value_current = target[action]
        
        if done:
            action_value_target = reward  
        else:
            next_state_active_actions  = active_network.predict(new_batched_state)[0]
            next_state_active_max_action = np.argmax(next_state_active_actions)
            target_next_state_active_action = target_network.predict(new_batched_state)[0, next_state_active_max_action]
            action_value_target = reward + gamma*target_next_state_active_action

            
        target[action] = action_value_target
        error = np.abs(action_value_current - action_value_target)
        
        x_batch_syndrome[counter,:] = batched_state[0][0,:]
        x_batch_actions[counter,:] = batched_state[1][0,:]
        counter += 1
        
        y_batch.append(target)
        errors.append(error)
        
    x_batch = [x_batch_syndrome,x_batch_actions]
        
    return x_batch, y_batch, errors

def check_early_stopping(episode_lengths,avg_episodes, stopping_patience, stopping_counter, previous_best):
    
    if len(episode_lengths) < avg_episodes + 1:
        return False, 0, 0
    
    new_avg = np.average(episode_lengths[-avg_episodes:])
    
    if new_avg > previous_best:
        previous_best = new_avg
        stopping_counter = 0
    else:
        stopping_counter +=1
    
    do_stop = False
    if stopping_counter >= stopping_patience:
        do_stop = True
        
    return do_stop, stopping_counter, previous_best

def update_epsilon(current_eps,eps_thresholds,eps_decays):
    eps_decay = 1 #In case no theshold is met, do not decay
    for j in range(np.shape(eps_thresholds)[0]):
        if current_eps > eps_thresholds[j]:
            eps_decay = eps_decays[j]
            break
    return current_eps*eps_decay

def train_agent(error_model,d,p_phys,qubits,num_actions,num_outputs,num_features,network_architecture,learning_rate,
               target_update_interval,min_buffer_size,max_buffer_size,batch_size,max_episodes,max_episode_length,
               stopping_patience,avg_episodes,gamma, max_simultaneous_moves, epsilon_initial,eps_thresholds, eps_decays,
               identity_percent,temp_initial, temp_decay, sample_dropout_initial, sample_dropout_min, 
                sample_dropout_decay, exploration, storage_interval,FFNN, file_path, architecture_string):
    
    # Initialize the memory_buffer
    memory_buffer = Memory(capacity=max_buffer_size)

    # Initialize lists to store improvement metrics
    episode_lengths = []
    epsilon_store = [epsilon_initial]
    temp_store = [temp_initial]
    dropout_store = [sample_dropout_initial]

    # 2) Initialize the Q network and the target Q network and the boltzmann model

    Q_ffnn = build_FF_DQN_with_streams(network_architecture, num_features, num_outputs, learning_rate)
    Q_ffnn_target = build_FF_DQN_with_streams(network_architecture, num_features, num_outputs, learning_rate)

    Q_ffnn_weights = Q_ffnn.get_weights()
    Q_ffnn_target.set_weights(Q_ffnn_weights)

    boltzmann_model = build_boltzmann_model(num_actions)

    # 3) Initialize relevant variables and counters

    stopping_counter = 0
    previous_best = 0
    previous_best_episode = 0
    stop_training = False
    total_updates_counter = 1
    epsilon = epsilon_initial
    temp = temp_initial
    sample_dropout = sample_dropout_initial
    total_moves_counter = 0
    print_flag = False


    episode = 1
    while not stop_training:

        stop_episode = False
        episode_length = 1

        episode_storage = []

        while not stop_episode and not stop_training:

          # --------- The episode always starts with the generation of a random error -------------

            if episode_length == 1:

                # Generate the initial error
                hidden_state = generate_X_error(d,p_phys)

                # Generate the intial syndrome
                new_visible_syndrome_state = generate_surface_code_syndrome_NoFT_efficient(hidden_state,qubits)
                new_visible_syndrome_state_vec = np.reshape(new_visible_syndrome_state,(d+1)**2)

                # Create a vector of the actions since the last error:
                completed_actions = np.zeros(num_actions)

                # Create the initial batched state
                batched_syndrome_state = np.expand_dims(new_visible_syndrome_state_vec,axis=0)
                batched_action_state = np.expand_dims(completed_actions,axis=0)
                batched_state = [batched_syndrome_state,batched_action_state]


            # ------ Choose an action via a given exploration strategy --------------------------

            if exploration == "e-greedy":
                action,action_index = epsilon_greedy_action(Q_ffnn,batched_state,d,epsilon,num_actions, 
                                                            identity_percent,error_model)

                # anneal the exploration probability
                epsilon = update_epsilon(epsilon,eps_thresholds,eps_decays)
                epsilon_store.append(epsilon)

            elif exploration == "boltzmann":
                action, action_index = boltzmann_action(boltzmann_model, Q_ffnn, temp, batched_state, 
                                                        num_actions,d,error_model)

                # anneal the temperature
                if temp != 1:
                    if temp > 1:
                        temp = temp*temp_decay
                    if temp < 1:
                        temp = 1
                temp_store.append(temp)

            elif exploration == "bayesian":
                action, action_index = bayesian_action(Q_ffnn, network_architecture, 
                                                       sample_dropout, batched_state, d, error_model,
                                                       num_features,num_outputs, learning_rate)

                # anneal the sample dropout
                if sample_dropout != sample_dropout_min:
                    if sample_dropout > sample_dropout_min:
                        sample_dropout = sample_dropout*sample_dropout_decay
                    if sample_dropout < sample_dropout_min:
                        sample_dropout = sample_dropout_min
                dropout_store.append(sample_dropout)

            # ------------ Act on the environment with the chosen action --------------

            new_hidden_state, new_visible_state, reward, terminal_state, new_syndrome = X_surface_code_environment_delayed_syndrome(
                qubits, hidden_state,  action, episode_length, max_simultaneous_moves, p_phys, FFNN)

            #---------- Create the memory/experience tuple for storage ------------

            if new_syndrome:
                # If we are at a time step where we should receive a new syndrome, then update this accordingly
                new_visible_syndrome_state_vec = np.reshape(new_visible_state,(d+1)**2)
                new_completed_actions = np.zeros(num_actions) 
            else:
                # If we are not at a time step where we do a syndrome measurement, then just update the action conditioning
                new_completed_actions = copy(completed_actions)
                if new_completed_actions[action_index] == 0:
                    new_completed_actions[action_index] = 1
                else:
                    new_completed_actions[action_index] = 0

            new_batched_syndrome_state = np.expand_dims(new_visible_syndrome_state_vec,axis=0)
            new_batched_action_state = np.expand_dims(new_completed_actions,axis=0)
            new_batched_state = [new_batched_syndrome_state,new_batched_action_state]

            current_memory = [batched_state,action_index,reward,new_batched_state,terminal_state]
            x_b,y_b,errors = construct_targets_from_batch_double_Q([(0,current_memory)], 
                                                                   Q_ffnn_target, Q_ffnn, gamma, d, num_actions)
            memory_buffer.add(errors[0], current_memory)
            total_moves_counter += 1

            batched_state = new_batched_state
            hidden_state = new_hidden_state
            completed_actions = new_completed_actions

            # ---- Train the neural network ---------------------------------------------

            if total_moves_counter > min_buffer_size:

                # Fetch a prioritized sample
                batch = memory_buffer.sample(batch_size)

                # Get the training data
                x_b,y_b,errors = construct_targets_from_batch_double_Q(batch, Q_ffnn_target, Q_ffnn, gamma,d,num_actions)

                # update the errors
                for i in range(len(batch)):
                    idx = batch[i][0]
                    memory_buffer.update(idx, errors[i])

                # Fit the neural network on this batch
                Q_ffnn.fit(x=x_b, y = np.array(y_b), batch_size=batch_size,verbose=0)

                # If necesary, update the target network
                if total_updates_counter % target_update_interval == 0:
                    active_weights = Q_ffnn.get_weights()
                    Q_ffnn_target.set_weights(active_weights)


                total_updates_counter +=1

            # ------ Check if the episode should end ---------------------------------


            # check if the episode should end:                                                      
            if terminal_state or episode_length > max_episode_length:
                stop_episode = True

                episode_lengths.append(episode_length)

                if episode % storage_interval == 0:
                    print_flag = True
                    # Save the Neural Network
                    Q_ffnn.save(file_path + "DQN_"+architecture_string+"_after_episode_"+str(episode)+".h5")
                    # Save the episode lengths
                    pickle.dump(episode_lengths, open(file_path+architecture_string+"_episode_lengths_"+str(episode)+".p", "wb" ) )

                # Check  if early stopping of training should occur  
                stop_training, stopping_counter, previous_best= check_early_stopping(episode_lengths,avg_episodes, 
                                                                       stopping_patience, stopping_counter, previous_best)

                if stopping_counter == 0:
                    previous_best_episode = episode


                print("Results From Episode Number: ", episode)
                print()
                print("Episode Length: ", episode_length)
                print("Rolling Average: ", np.average(episode_lengths[-avg_episodes:]))
                print("Previous Best: ", previous_best)
                print("Previous Best Episode: ", previous_best_episode)
                if print_flag:
                    print("--- Saving Results -----")
                    print_flag = False
                print()

                episode +=1

            # check if training should end
            if stop_training or episode > max_episodes:
                stop_training = True
                print(" ------ ")
                print("Training Stopped")
                episode_lengths.append(episode_length)
                
                # Save the Neural Network
                Q_ffnn.save(file_path + "DQN_"+architecture_string+"_after_episode_"+str(episode)+".h5")
                # Save the episode lengths
                pickle.dump(episode_lengths, open(file_path+architecture_string+"_episode_lengths_"+str(episode)+".p", "wb" ) )


            episode_length += 1


# ------------------- Visual Debugging Tools ------------------

def draw_code(num_qubits, image_width=300):
    num_blocks = num_qubits + 1
    block_width = image_width/num_blocks
    
    my_image = im.new("RGB",size=(image_width,image_width),color="yellow")
    draw = imdraw.Draw(my_image)
    black = True
    for j in range(1,num_blocks-1):
        for k in range(1,num_blocks-1):
            tlx = j*block_width
            tly = k*block_width
            brx = tlx + block_width
            bry = tly + block_width
            if j != 0 and j != num_blocks - 1 and j%2 != 0:
                if k != 0 and k != num_blocks and k%2 != 0:
                    draw.rectangle((tlx,tly,brx,bry),"black","black")
                elif k != 0 and k != num_blocks:
                    draw.rectangle((tlx,tly,brx,bry),"white","black")
            else:
                if k != 0 and k != num_blocks and k%2 == 0:
                    draw.rectangle((tlx,tly,brx,bry),"black","black")
                elif k != 0 and k != num_blocks:
                    draw.rectangle((tlx,tly,brx,bry),"white","black")
            if j == 0:
                draw.pieslice((tlx,tly+block_width/2,brx,bry+block_width/2),180,0,"black")         

    j = 0
    for k in [2*x for x in range(1,num_blocks-1)]:
            tlx = j*block_width
            tly = k*block_width
            brx = tlx + block_width
            bry = tly + block_width
            draw.pieslice((tlx+block_width/2,tly,brx+block_width/2,bry),90,270,"black")   

    j = num_blocks -1
    for k in [2*x -1 for x in range(1,int(num_blocks/2))]:
            tlx = j*block_width
            tly = k*block_width
            brx = tlx + block_width
            bry = tly + block_width
            draw.pieslice((tlx-block_width/2,tly,brx-block_width/2,bry),270,90,"black")   

    k = 0
    for j in [2*x -1 for x in range(1,int(num_blocks/2))]:
            tlx = j*block_width
            tly = k*block_width
            brx = tlx + block_width
            bry = tly + block_width
            draw.pieslice((tlx,tly+block_width/2,brx,bry+block_width/2),180,0,"white","black")

    k = num_blocks -1 
    for j in [2*x for x in range(1,num_blocks-1)]:
            tlx = j*block_width
            tly = k*block_width
            brx = tlx + block_width
            bry = tly + block_width
            draw.pieslice((tlx,tly-block_width/2,brx,bry-block_width/2),0,180,"white","black")

    return my_image, block_width

def add_qubit_flip(current_image, qubit_row, qubit_column, block_width, pauli_basis):
    if pauli_basis != 0:
        if pauli_basis == 1:
            message = "X"
        elif pauli_basis == 2:
            message = "Y"
        elif pauli_basis == 3:
            message = "Z"
            
        draw = imdraw.Draw(current_image)
        centre_x = (qubit_column + 1)*block_width
        centre_y = (qubit_row+1)*block_width
        fl = 0.15*block_width

        draw = imdraw.Draw(current_image)
        draw.rectangle((centre_x-fl,centre_y-fl,centre_x+fl,centre_y+fl),"blue","red")

        font=ImageFont.truetype("arial",18)
        width, hight = draw.textsize(message, font=font)
        draw.text((centre_x-width/2,centre_y-hight/2),message,fill=(255,255,255,255),font=font)
    return current_image

def add_anyon(current_image, stabilizer_row, stabilizer_column, block_width, num_qubits):
    draw = imdraw.Draw(current_image)
    
    if stabilizer_row not in [0,num_qubits] and stabilizer_column not in [0,num_qubits]:
        tlx = stabilizer_column*block_width
        tly = stabilizer_row*block_width
        brx = tlx + block_width
        bry = tly + block_width
        r = 0.30*block_width
        draw.ellipse((tlx+r,tly+r,brx-r,bry-r),"red","blue")
    elif stabilizer_row == 0:
        tlx = stabilizer_column*block_width
        tly = stabilizer_row*block_width + 0.252*block_width
        brx = tlx + block_width
        bry = tly + block_width
        r = 0.30*block_width
        draw.ellipse((tlx+r,tly+r,brx-r,bry-r),"red","blue")
    elif stabilizer_row == num_qubits:
        tlx = stabilizer_column*block_width
        tly = stabilizer_row*block_width - 0.252*block_width
        brx = tlx + block_width
        bry = tly + block_width
        r = 0.30*block_width
        draw.ellipse((tlx+r,tly+r,brx-r,bry-r),"red","blue")
    elif stabilizer_column == num_qubits:
        tlx = stabilizer_column*block_width - 0.252*block_width
        tly = stabilizer_row*block_width 
        brx = tlx + block_width
        bry = tly + block_width
        r = 0.30*block_width
        draw.ellipse((tlx+r,tly+r,brx-r,bry-r),"red","blue")
    elif stabilizer_column == 0:
        tlx = stabilizer_column*block_width + 0.252*block_width
        tly = stabilizer_row*block_width 
        brx = tlx + block_width
        bry = tly + block_width
        r = 0.30*block_width
        draw.ellipse((tlx+r,tly+r,brx-r,bry-r),"red","blue")
        
    return current_image

def draw_errors(current_image, hidden_state, block_width):
    num_qubits = np.shape(hidden_state)[0]
    for qubit_row in range(num_qubits):
        for qubit_column in range(num_qubits):
            current_image = add_qubit_flip(current_image,qubit_row,qubit_column,block_width,hidden_state[qubit_row,qubit_column])

    return current_image

def draw_syndrome(current_image, syndrome, block_width):
    num_qubits = np.shape(syndrome)[0] - 1
    for stabilizer_row in range(num_qubits+1):
        for stabilizer_column in range(num_qubits+1):
            if syndrome[stabilizer_row,stabilizer_column] == 1:
                current_image = add_anyon(current_image, stabilizer_row, stabilizer_column, block_width, num_qubits)
            
    return current_image

def draw_code_state(hidden_state,image_width=300):
    my_image = []
    num_qubits = np.shape(hidden_state)[0]
    qubits = generateSurfaceCodeLattice(num_qubits)
    lattice, block_width = draw_code(num_qubits,image_width)
    lattice = draw_errors(lattice,hidden_state, block_width)
    syndrome =  generate_surface_code_syndrome_NoFT_efficient(hidden_state,qubits)
    lattice = draw_syndrome(lattice, syndrome,block_width)
    return lattice

def concatenate_images(images):
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = pil.Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    return new_im

def concatenate_images_vertically(images):
    widths, heights = zip(*(i.size for i in images))

    total_height = sum(heights)
    max_width = max(widths)

    new_im = pil.Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0,y_offset))
        y_offset += im.size[1]
    return new_im

def show_state_tuple(state_tuple, d, error_model, use_Y=True, image_width=200):
    syndrome = np.reshape(state_tuple[0],(d+1,d+1))
    actions = state_tuple[1]
    
    base_lattice_s, block_width = draw_code(d,image_width)
    invisible_syndrome_fig = draw_syndrome(base_lattice_s, syndrome, block_width)
    
    action_lattice = convert_completed_actions_to_lattice(actions,d,error_model,use_Y)
    base_lattice_a, block_width = draw_code(d,image_width)
    action_im = draw_errors(base_lattice_a, action_lattice, block_width)
    
    return concatenate_images([invisible_syndrome_fig, action_im])

# THERE ARE ERRORS IN THIS FUNCTION!
def convert_completed_actions_to_lattice(completed_act,d,error_model,use_Y=True):
    base_lattice = np.zeros((d,d))
    for count,val in enumerate(list(completed_act)):
        if val == 1.0:
            move =  index_to_move(d,count,error_model,use_Y)
            base_lattice = obtain_new_error_configuration(base_lattice, move)
            
    return base_lattice

def show_history(memory,d, error_model,use_Y=True, image_width=200):
    
    image_list = []
    for state_tuple in memory:
        image_list.append(show_state_tuple(state_tuple,d,error_model,use_Y,image_width))
    
    return concatenate_images_vertically(image_list)

def show_volume(error_list,volume_hidden_states,volume_true_syndromes, volume_faulty_syndromes, visible_volume, d, image_width=200):

    volume_depth = len(error_list)
    volume_slices =[]
    for j in range(volume_depth):

        base_lattice_e, block_width = draw_code(d,image_width)
        e_im = draw_errors(base_lattice_e, error_list[j], block_width)

        h_im = draw_code_state(volume_hidden_states[j], image_width)

        base_lattice_ts, block_width = draw_code(d,image_width)
        ts_im = draw_syndrome(base_lattice_ts, volume_true_syndromes[j], block_width)

        base_lattice_fs, block_width = draw_code(d,image_width)
        fs_im = draw_syndrome(base_lattice_fs, volume_faulty_syndromes[j], block_width)

        base_lattice_vs, block_width = draw_code(d,image_width)
        vs_im = draw_syndrome(base_lattice_vs, visible_volume[j], block_width)

        time_slice = [e_im, h_im, ts_im, fs_im, vs_im]
        time_slice_im = concatenate_images(time_slice)
        volume_slices.append(time_slice_im)

    return concatenate_images_vertically(volume_slices)





