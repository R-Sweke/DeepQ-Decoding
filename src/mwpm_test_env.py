from deepq.Environments import *
from deepq.Function_Library import *
from deepq.Utils import *

import matplotlib.pyplot as plt

# environment parameter
d=3
error_model='DP'

# create the environment with MWPM referee decoder
# by simply not providing any static decoder.
env = Surface_Code_Environment_Multi_Decoding_Cycles(
    d=d, 
    p_phys=0.003, 
    p_meas=0.00,  
    error_model=error_model, 
    use_Y=False, 
    volume_depth=d)

# reset environment and generate first syndrome error volume
obs = env.reset()
env.step(1)