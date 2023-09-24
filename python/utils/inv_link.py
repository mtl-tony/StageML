import numpy as np

# Identity link 
def inv_identity(x):
    return x

# Log link
def inv_log(x):
    return np.exp(x)

# Inverse Logistic
def inv_logistic(x):
    return np.log(x / (1 - x))