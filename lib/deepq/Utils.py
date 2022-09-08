from enum import Enum
import numpy as np
from typing import List

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from pymatching import Matching
from scipy.sparse import csc_matrix

class MatchingDecoder():
  """
  MWPM decoder based on PyMatching.
  """
  
  def __init__(self, parity_check_matrices: List[np.array]):
    """
    Decoder accepts list of parity check matrices. Unfortunately those
    have to be in order, meaning first X then Z matrix.
    """
    self.parity_check_matrices = parity_check_matrices
    self.M = []
    
    for H in self.parity_check_matrices:
      self.M.append(self.get_matching_graph(H))


  def get_matching_graph(self, H: np.array) -> Matching:
    """
    Returns matching graph for a given parity check matrix

    :param: H: stabilizer parity check matrix
    """

    matching = Matching()
    num_syndromes = H.shape[0] # rows correspond to syndromes
    num_qubits = H.shape[1] # columns correspond to data qubits
    singletons = len(np.where(np.sum(H, axis=0) == 1)[0]) # num data qubits supported by only one stabilizer
    boundary_indices = [e + num_syndromes for e in range(singletons)] # indices for boundary nodes
    weights = np.ones(num_qubits+1) # Manhattan distance
    error_probabilities = np.ones(num_qubits+1) # Equi-probable errors

    # csr matrix explained: https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr
    # note: matrix below is CSC!
    H_sparse = csc_matrix(H)
    bound_indices_iterator = iter(boundary_indices)

    for j in range(len(H_sparse.indptr) - 1):
      s, e = H_sparse.indptr[j:j + 2] # give me the range in which data for column j is
      v1 = H_sparse.indices[s] # in which row do we place first data
      # check if v2 has to be a boundary node
      if e - s == 1:
        v2 = next(bound_indices_iterator)
      else: # not a boundary node since supported by more than one plaquette/stabilizer
        v2 = H_sparse.indices[e - 1]
      matching.add_edge(v1, v2, fault_ids={j}, weight=weights[j],
                                error_probability=error_probabilities[j])
    
    # connect all boundary nodes with edge weight 0.
    # this allows to take shortcut via boundaries and therefore correct less qubits
    # by taking a short path over boundary instead of through the code.
    for i in boundary_indices:
      for j in range(i+1, num_syndromes+singletons):
        if i != j:
          matching.add_edge(i,j, weight=0, fault_ids=-1, error_probability=0)
    
    matching.set_boundary_nodes(set(boundary_indices))
    return matching

  def predict(self, syndromes: List[np.array]) -> List[np.array]:
    corrections = []
    for idx, syn in enumerate(syndromes):
      # set num_neighbours to None to ensure exact matching in PyMatching
      corrections.append(self.M[idx].decode(syn, num_neighbours=None))
    return corrections

def get_parity_matrix(stab_list, syndromes, pauli: int, d: int) -> np.array:
  """
  Returns the parity matrix for a list of stabilizers per qubit

  :param: stab_list: for each qubit list of stabilizer coordinates
  :param: syndromes: list of stabilizers for each data qubit
  :param: pauli: Pauli operator (I, X, Y, Z)
  :param: d: surface code distance
  """
  num_pauli_stabs = (d**2-1)//2
  H = np.zeros((num_pauli_stabs, d**2),int)
  for idx, qubit_stab_list in enumerate(stab_list):
    for stab in qubit_stab_list:
      if syndromes[stab[0], stab[1], 0] == pauli:
        stab_idx = syndromes[stab[0], stab[1], 1]
        H[stab_idx][idx] = 1
  return H


def get_syndrome_vector(syn, syndromes, pauli: int, d: int) -> np.array:
  """
  Returns a vector of syndromes for a given Pauli operator

  :param: syn:
  :param: syndromes:
  :param: pauli: Pauli operator (I, X, Y, Z)
  :param: d: surface code distance
  """
  syn_vec = np.zeros((d**2-1)//2)
  indices = np.where(syn == 1)
  for i, j in zip(indices[0], indices[1]):
    if syndromes[i,j][0] == pauli:
      idx = syndromes[i,j][1]
      syn_vec[idx] = 1
  return syn_vec

def draw_surface_code(state, syndromes, measured_syndromes, d, corrections=None):
  """"
  Method to draw surface code with current data qubit, ancilla qubit states
  
  :param: state: state of data qubits
  :param: syndromes: numpy array indicating type of plaquette
  :param: measured_syndromes: a syndrome volume
  :param: d: surface code distance
  :param: corrections: qubit coordinates selected for correction by decoder
  """

  fig = plt.figure()
  ax = fig.add_subplot(111)
  # only integers on axes
  ax.yaxis.get_major_locator().set_params(integer=True)
  ax.xaxis.get_major_locator().set_params(integer=True)
  # blueish : X, reddish: Z plaquette
  syn_colors={1: '#CFCCF9', 3: '#FECCCB'}
  err_colors={0: 'black', 1: 'yellow', 2: 'red', 3: 'green'}

  custom_lines = [Line2D([0], [0], color='yellow', lw=4),
                  Line2D([0], [0], color='red', lw=4),
                  Line2D([0], [0], color='green', lw=4)]

  for i in range(d+1):
    for j in range(d+1):
      # draw syndromes plaquettes
      pauli = syndromes[i,j,0]
      if pauli in [1,3]:
        r=Rectangle(xy=(j,i), width=1, height=1, color=syn_colors[pauli])
        ax.add_patch(r)

        # draw error markers on syndromes
        if measured_syndromes[i,j] != 0:
          r=plt.Circle(xy=(j+0.5, i+0.5), radius=0.2, color=err_colors[pauli])
          ax.add_patch(r)

      # draw data qubits and indicate error
      if i < d and j < d:
        plt.plot(j+1,i+1, marker='o', color=err_colors[state[i,j]])

    if corrections is not None:
      for (i,j) in corrections:
        plt.plot(j+1,i+1, marker='o', color='magenta')

  ax.legend(custom_lines, ['X', 'Y', 'Z'])
  plt.gca().set_aspect('equal', adjustable='box')
  plt.xlim([0,d+1])
  plt.ylim([0,d+1])
  plt.gca().invert_yaxis()


