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
    Return matching graph for a given parity check matrix

    :param: H: stabilizer parity check matrix
    """

    matching = Matching()
    num_detectors = H.shape[0]
    num_qubits = H.shape[1]
    singletons = len(np.where(np.sum(H, axis=0) == 1)[0])  
    boundary = [e + num_detectors for e in range(singletons)]
    weights = np.ones(num_qubits+1)
    error_probabilities = np.ones(num_qubits+1)

    H = csc_matrix(H)
    bound_iterator = iter(boundary)
    for i in range(len(H.indptr) - 1):
      s, e = H.indptr[i:i + 2]
      v1 = H.indices[s]
      v2 = H.indices[e - 1] if e - s == 2 else next(bound_iterator)
      matching.add_edge(v1, v2, fault_ids={i}, weight=weights[i],
                                error_probability=error_probabilities[i])
    matching.set_boundary_nodes(set(boundary))

    return matching

  def predict(self, syndromes: List[np.array]) -> List[np.array]:
    corrections = []
    for idx, syn in enumerate(syndromes):
      corrections.append(self.M[idx].decode(syn))
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


