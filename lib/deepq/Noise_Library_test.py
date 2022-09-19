import numpy as np

from deepq.Noise_Library import XNoise

class TestNoiseLibrary:

  def test_xnoise_10000_samples(self):
    d = 7
    p_phys = 0.01
    x_noise = XNoise(d, p_phys)
    samples = 10000
    
    board = np.zeros((d,d),int)
    for _ in range(samples):
        board += x_noise.generate_error()

    # TODO : use confidence interval
    empirical_mean = np.mean(board)
    assert(np.abs(empirical_mean - samples*p_phys) < 20)
    