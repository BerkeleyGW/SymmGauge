import numpy as np

# Fermi level
nfer = 19
nfer_reduce_fi = 3  # since we are using cut version of wfn_fi

# Bands we want to focus
# nb = 3
# nc = 1

nv_fi = 3
nc_fi = 1

bstart_fi = nfer_reduce_fi - nv_fi
bend_fi = nfer_reduce_fi + nc_fi


# Bnads at hight symmetry kpoint we want to take as a reference

nv_co_k0 = 10
nc_co_k0 = 10


# Index of hight symmetry k point
idx_k0 = 0

idx_fi_k0 = 5469

spinor = False

# valence bands are reversed
v_rever = True

# file name
co_name = "wfn_co.h5"
fi_name = "wfn_co.h5"

use_ngk = True

gcut = 20000


# max error : 6.052424282696001e-08 without gvec pools
# dn0n[:, param.nv_co_k0:param.nv_co_k0+1 , 3:4 ] -  di_c = dcn[:, 0:1, :]  5.4453811228041395e-08


# For similarity transformation
irlabel = "G3"

op_c3 = 1
op_m100 = 4
dg = [1,2,1]
db_s = 1
db_e = 3
