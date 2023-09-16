#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
8/7/2021
Author: jwruan
input: wfn_co.h5, wfn-allk.h5

output:
dnc_10co_2fi-check.npy
======
Created on Sat Dec 12 22:21:19 2020

@author: jwruan
"""



import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys

co_name = "wfn_co.h5"
fi_name = "wfn_fi.h5"
f_co = h5py.File(co_name)
# f_fi = h5py.File(fi_name)

f_fi = h5py.File("wfn-allk.h5")

nk_co = 2
nk_fi = 4341
ind_kc = 2170

gcut = 43556
tmp = f_co['wfns/coeffs'][()]
# look = tmp
tmp = tmp[:,:,:,0] + 1j*tmp[:,:,:,1]
tmp = tmp.reshape(250, 2, nk_co, -1 )   # 250: dimension for co bands
coeff_co = tmp.transpose(2,0,1,3)
coeff_co = coeff_co[:,:,:,:gcut]      # nk x nb x nspinor x cplex


print("\nread fine wavefunction")
tmp = f_fi['wfns/coeffs'][()]
tmp = tmp[:,:,:,0] + 1j*tmp[:,:,:,1]
tmp = tmp.reshape(4, 2, nk_fi, -1 )   # 4: dimension for fine bands
coeff_fi = tmp.transpose(2,0,1,3)
coeff_fi = coeff_fi[:,:,:,:gcut]   # nk x nb x nspinor x cplex

print("end reading fine wfc ")


def dotspinor(v1,v2):
    # dot product for spinor wavefucntion
    result = v1[0,:].conj().dot(v2[0,:]) + v1[1,:].conj().dot(v2[1,:])
    # print(result)
    return result


bstart_co = 58 -10
bend_co =  58 + 10
nbnd = bend_co - bstart_co

nferi = 2
nbc_fi = 2
nbv_fi = 1
dnc = np.zeros((nk_fi,nbnd, nbc_fi ), dtype='complex128')   # nk x co x fi
my_dvn = np.zeros((nk_fi,nbnd,nbv_fi ), dtype='complex128')


print("\nDo the dot product for dcn")

for k in range(nk_fi):
    # my_dvn[k,0,0] = dotspinor(coeff_co[0,57], coeff_fi[k,57])
    for i, idx in enumerate(list(range(bstart_co, bend_co))):
        for j in range(nbc_fi):
            dnc[k,i,j] = dotspinor(coeff_co[0,idx], coeff_fi[k,nferi+j])   # 0 is correct, becase we have only two k points in coeff_co


# ADDED later
np.save("dnc_10co_2fi-check.npy", dnc)

print("\nEnd dot product for dcn")

f_co.close()
f_fi.close()



#======================== end heavy reading ========================

# f_vmt = h5py.File('vmtxel-morebands.h5')
# vmtxel_data= f_vmt['vmtxel_data/dipole'][()]   # npol x s x k x v xc

# dcn_dtmat = np.load('dcn_dtmat.npy')
# dvn_dtmat = np.load('dvn_dtmat.npy')

# dtrans = np.load('dtrans_regular.npy')     # nk x nc_co x nc_fi
container = np.load('dtrans_regular_similarity.npz')
dtrans = container['dtrans']
smat = container['smat']      # < dft_co | reg_co >


my_dcc = dnc[:, 10: 12, :]    # We can refer it from bstart_co, bend_co
dcc_reg = np.einsum("ij,kjf->kif", np.linalg.inv(smat), my_dcc)  # < reg_co_zero | fi >



# dcc_reg_norm =

# # copy from MMA
# posL2 = -1 + np.array([2171, 2310, 2449, 2586, 2722, 2857, 2990, 3122, 3250, 3374, 3494, \
# 3611, 3723, 3831, 3934, 4029, 4116, 4195, 4263, 4315])
# posL1 = -1 + np.array([2171, 2242, 2313, 2384, 2455, 2525, 2595, 2665, 2734, 2803, 2872, \
# 2940, 3008, 3076, 3143, 3209, 3274, 3338, 3401, 3463, 3524])

# dtrans_L2 = dtrans[posL2]
# sum_L2 = [dtrans_L2[i].dot(dtrans_L2[i].conj().T) for i in range(len(dtrans_L2))]
# sum_L2 = np.array(sum_L2)


# def magicrot(dcn):
#     tmpsum = [dcn[i].dot(dcn[i].conj().T) for i in range(len(dcn))]
#     tmpsum = np.array(tmpsum)
#     return tmpsum


# magicrot(dcn_dtmat[posL2])


# data = np.linalg.norm(dtrans_L2[:,:,0],axis=1)**2
# # plt.plot(data)
# # plt.ylim(0.0, 1.0)
# # plt.show()
