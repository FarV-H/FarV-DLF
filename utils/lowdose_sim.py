import numpy as np

def LowDoseSimulate(sino_label, I0):
    Ne          = 8.2

    I_out       = I0 * np.exp(-sino_label)
    sigma       = I_out + Ne
    I_out       = I_out + np.sqrt(sigma) * np.random.randn(*I_out.shape)
    u           = I_out / I0
    u[u<1e-07]  = 1e-07
    u[u>10]     = 10
    sino_ld     = -np.log(u)
    del Ne, I_out, sigma, u

    return sino_ld