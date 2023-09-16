#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 20:34:07 2020

@author: jwruan

input: 
    dnc_10co_2fi.npy, 
    dtrans_regular_similarity.npz
    dnc_10co_2fi.npy            # overlap matrix. Raw data
"""

# import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm

dn0c_dft = np.load("dnc_10co_2fi.npy")  # nk x (nv+nc) x nb_fi, here 0 means <u(r)k=0 |


container = np.load('dtrans_regular_similarity.npz')
# dtrans = container['dtrans']
smat = container['smat']      # < dft_co | reg_co >

dc0c_dft = dn0c_dft[:, 10: 12, :]    # We can refer it from bstart_co, bend_co

dc0c = np.einsum("ij,kjf->kif", np.linalg.inv(smat), dc0c_dft)  # < reg_co_zero | fi >

# change the dn0c from raw data to have regular one
dn0c = dn0c_dft.copy()
dn0c[:, 10:12, :] = dc0c

# copy from MMA
posL2 = -1 + np.array([2171, 2310, 2449, 2586, 2722, 2857, 2990, 3122, 3250, 3374, 3494, \
3611, 3723, 3831, 3934, 4029, 4116, 4195, 4263, 4315])
posL1 = -1 + np.array([2171, 2242, 2313, 2384, 2455, 2525, 2595, 2665, 2734, 2803, 2872, \
2940, 3008, 3076, 3143, 3209, 3274, 3338, 3401, 3463, 3524])


def get_hermi(dc0c):  #  U.U_dag, input: nk x nb_co0 x nb_fi
    tmpsum = [dc0c[i].dot(dc0c[i].conj().T) for i in range(len(dc0c))]
    tmpsum = np.array(tmpsum)
    return tmpsum


dc0c_squre = get_hermi(dc0c)
dc0_chic = np.array([sqrtm(m) for m in dc0c_squre])  # < phi | chi >  # squre root of matrix
Du2_chnel = np.array([np.linalg.inv(v).dot(u) for u, v in zip(dc0c, dc0_chic) ])  # inv(<phi|chi>).<phi| psi> = <chi|psi>
print("\n\n Save Du2_chnel\n\n")
np.save("Du2_chnel", Du2_chnel)

dn0_chic = np.array([u.dot(np.linalg.inv(v)) for u, v in zip(dn0c, Du2_chnel)])     # <phi|psi> U^-1 = <phi | chi> for each k point

np.save('dn0_chic.npy', dn0_chic) # nk x nn0 x nc

print("Finished")
