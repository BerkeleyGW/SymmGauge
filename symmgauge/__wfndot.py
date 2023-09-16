#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
8/7/2021
Author: jwruan
input: wfn_co.h5, wfn-allk.h5

output:
dnc_10co_2fi-check.npy
======

@author: ruanjiaw@gmail.com
"""


import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import sys

# fname = 'wfn-allk.h5'
fname = 'wfn_co.h5'
ind_kc = 2170

# def

with h5.File(fname, 'r') as f:
    tmp = f.get('wfns/coeffs')[()]
    tmp = tmp[:,:,:,0] + 1j*tmp[:,:,:,1]   # nb x nspinor x (ngktot)
    nrk = f.get('mf_header/kpoints/nrk')[()]
    nspinor = f.get('mf_header/kpoints/nspinor')[()]
    mnband = f.get('mf_header/kpoints/mnband')[()]
    ngk = f.get('mf_header/kpoints/ngk')[()]
    ngkmax = f.get('mf_header/kpoints/ngkmax')[()]

    ifmax = f.get('mf_header/kpoints/ifmax')[()]

    coeff_fi = tmp.reshape(mnband, nspinor, nrk, ngkmax)
    coeff_fi = coeff_fi.transpose(2,0,1,3)   # nk x nb x nspinor x ngevc

    # tmp = None



# end def
