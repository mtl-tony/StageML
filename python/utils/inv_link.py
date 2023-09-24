import numpy as np

# Identity link 
def link_identity(x):
    return x
def inv_identity(x):
    return x

# Log link
def link_log(x):
    return np.log(x)
def inv_log(x):
    return np.exp(x)


# Inverse Logistic
def link_logistic(x):
    return np.log(x / (1 - x))
def inv_logistic(x):
    return 1 / (1 + np.exp(-x))