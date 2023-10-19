import h5py
import sys
import numpy as np
from symmgauge.vmtxel import Vmtxel

from symmgauge.wfnio import  Wfn
from symmgauge.gauge import GaugeKp

fname1 = 'wfn.h5'
vmtseedname = '../data_bgw/eigenvalues'


nv_fi = 2     # be consistent with the bse calculation
nc_fi = 2
idx_k0 = 9

bpick_list = [[nv_fi-1],[nv_fi,nv_fi + 1]]
reg_list = [None, None]


wfntest = Wfn(fname1)

gfield = GaugeKp(wfntest, nv_fi, nc_fi, idx_k0 )

for bp, reg in zip(bpick_list, reg_list):
    dn0n = gfield.update_dn0n(bp, reg)   # dummpy update
    gauge = gfield.update_gauge(bp, dn0n)

hbnvmt = Vmtxel(vmtseedname=vmtseedname, mffile=fname1, gaugecls=gfield)

hbnvmt.cal_k0_trans(idx_k0, cinds=[0,1],vinds=[0])
