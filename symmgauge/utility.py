import numpy as np
from scipy.linalg import sqrtm



def svd_like(submat):
    '''
    get the gauge transformation matrix
    input: <phi | psi>
    '''
    dm0m_squre = submat.dot(submat.conj().T)
    dm0_chiv =sqrtm(dm0m_squre) # < phi | chi >  # squre root of matrix

    gauge_tran = np.linalg.inv(dm0_chiv).dot(submat) # inv(<phi|chi>).<phi|psi> = <chi|psi>

    return gauge_tran
