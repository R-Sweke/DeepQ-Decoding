import numpy as np

class NoiseFactory():

  def __init__(self, error_model, d, p_phys, **kwargs):
    self.error_model = error_model
    self.d = d
    self.p_phys = p_phys
    self.kwargs = kwargs

  def generate(self):
    if self.error_model == "X":
      return XNoise(self.d, self.p_phys)
    elif self.error_model == "DP":
      return DPNoise(self.d, self.p_phys)
    elif self.error_model == "IIDXZ":
      return IIDXZNoise(self.d, self.p_phys)
    elif self.error_model == "HEAT":
      if "p_X" in self.kwargs and "p_Z" in self.kwargs:
        heatmaps = [self.kwargs["p_X"], self.kwargs["p_Z"]]
        return HeatmapNoise(self.d, self.p_phys, heatmaps)
    else:
      raise ValueError(f"Error model: {self.error_model} doesn't exist!")


class DPNoise():
  def __init__(self, d, p_phys):
    self.d = d
    self.p_phys = p_phys

  def get_error_model(self):
    return "DP"

  def generate_error(self):
      """"
      This function generates an error configuration, via a single application of the depolarizing noise channel, on a square dxd lattice.
      
      :param: d: The code distance/lattice width and height (for surface/toric codes)
      :param: p_phys: The physical error rate.
      :return: error: The error configuration
      """

      error = np.zeros((self.d, self.d), int)
      for i in range(self.d):
          for j in range(self.d):
              p = 0
              if np.random.rand() < self.p_phys:
                  p = np.random.randint(1, 4)
                  error[i, j] = p

      return error


class XNoise():
  def __init__(self, d, p_phys):
    self.d = d
    self.p_phys = p_phys

  def get_error_model(self):
    return "X"

  def generate_X_error(self):
      """"
      This function generates an error configuration, via a single application of the bitflip noise channel, on a square dxd lattice.
      
      :param: d: The code distance/lattice width and height (for surface/toric codes)
      :param: p_phys: The physical error rate.
      :return: error: The error configuration
      """

      error = np.zeros((self.d, self.d), int)
      for i in range(self.d):
          for j in range(self.d):
              p = 0
              if np.random.rand() < self.p_phys:
                  error[i, j] = 1

      return error


class IIDXZNoise():

  def __init__(self, d, p_phys):
    self.d = d
    self.p_phys = p_phys

  def get_error_model(self):
    return "IIDXZ"

  def generate_IIDXZ_error(self):
      """"
      This function generates an error configuration, via a single application of the IIDXZ noise channel, on a square dxd lattice.
      
      :param: d: The code distance/lattice width and height (for surface/toric codes)
      :param: p_phys: The physical error rate.
      :return: error: The error configuration
      """

      error = np.zeros((self.d, self.d), int)
      for i in range(self.d):
          for j in range(self.d):
              X_err = False
              Z_err = False
              p = 0
              if np.random.rand() < self.p_phys:
                  X_err = True
                  p = 1
              if np.random.rand() < self.p_phys:
                  Z_err = True
                  p = 3
              if X_err and Z_err:
                  p = 2

              error[i, j] = p

      return error


class HeatmapNoise():

  def __init__(self, d, p_phys, p_X, p_Z):
    self.d = d
    self.p_phys = p_phys
    self.p_X = p_X
    self.p_Z = p_Z

  def get_error_model(self):
    return "HEAT"

  def generate_heatmap_error(self):
      """"
      This function generates an error configuration, using a dxd heatmap as input that servers as error probability distribution
      for every qubit.

      :param: d: The code distance/lattice width and height (for surface/toric codes)
      :param: heatmaps: List of Numpy arrays, shape dxd. [heatmap_X, heatmap_Z] is the default
      """

      error = np.zeros((self.d,self.d),int)
      for i in range(self.d):
          for j in range(self.d):
              X_err = False
              Z_err = False
              p = 0
              if np.random.rand() < self.p_X[i,j]:
                  X_err = True
                  p = 1
              if np.random.rand() < self.p_Z[i,j]:
                  Z_err = True
                  p = 3
              if X_err and Z_err:
                  p = 2

              error[i,j] = p
      
      return error