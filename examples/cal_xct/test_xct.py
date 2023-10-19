import h5py
import sys
import numpy as np
from symmgauge.vmtxel import Vmtxel
from symmgauge.xct import Xct

from symmgauge.wfnio import  Wfn
from symmgauge.gauge import GaugeKp

fname1 = 'wfn.h5'
fname_xct = 'eigenvectors.h5'


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

# enough for single band to single band transition
hbxct = Xct(fname=fname_xct, gaugecls=gfield)
