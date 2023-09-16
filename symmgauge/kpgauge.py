import h5py
import sys
import param
import numpy as np
import pickle
from similarity import AbstractG6_2, similarity, rep_G6_2
from scipy.linalg import sqrtm


def raw2reg(fname='repmat.pkl'):
    irlabel = param.irlabel


    with open(fname, 'rb') as f_in:
        data = pickle.load(f_in)

    k0_rep = data['mat']

    sliceobj = slice(param.db_s, param.db_e)
    d_c3 = k0_rep[param.op_c3][sliceobj,sliceobj]     # URGE
    d_m1 = k0_rep[param.op_m100][sliceobj, sliceobj]


    C3_regular = rep_G6_2[irlabel]['C3']
    M1_regular = rep_G6_2[irlabel]['M1']


    rep1 = AbstractG6_2(d_c3, d_m1)
    rep2 = AbstractG6_2(C3_regular, M1_regular)

    smat = similarity(rep1, rep2)

    dim = k0_rep.shape[-1]
    k0_trans = np.eye(dim, dtype=np.complex)

    k0_trans[sliceobj, sliceobj] = smat        # < dft_co | reg_co >

    return k0_trans


def get_hermi(dm0m):   #  U.U_dag, input: nk x nb_co0 x nb_fi
    tmpsum = [dm0m[i].dot(dm0m[i].conj().T) for i in range(len(dm0m))]
    tmpsum = np.array(tmpsum)
    return tmpsum

def get_gauge(dm0m):
    '''get the gauge transformation matrix  '''
    dm0m_squre = get_hermi(dm0m)
    dm0_chiv = np.array([sqrtm(m) for m in dm0m_squre])  # < phi | chi >  # squre root of matrix

    gauge_tran = np.array([np.linalg.inv(v).dot(u) for u, v in \
                           zip(dm0m, dm0_chiv) ])  # inv(<phi|chi>).<phi|psi> = <chi|psi>

    return gauge_tran



def check_psot_def(dm0m):
    tmp = np.array([ np.diagonal(i) for i in dm0m])
    return tmp

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


if __name__='__main__':

    slobj = slice(param.nv_co_k0-param.nv_fi, param.nv_co_k0 + param.nc_fi)

    dn0n_dft = np.load("dn0n.npy")      # nk x (nv + nc) x nb_fi, here 0 means <u(r) k = 0 |
    dm0m_dft = dn0n_dft[:, slobj, :]    # now we should have a square matrix

    k0_trans = raw2reg('repmat.pkl')

    dm0m = np.einsum("ij,kjf->kif", np.linalg.inv(k0_trans), dm0m_dft)       # <reg_co_zero | fi >
    dn0n = dn0n_dft.copy()
    dn0n[:, slobj, :] = dm0m              # put the square back. May used to study pseudo-spin physics

    dv0v = dm0m[:, :param.nv_fi, :param.nv_fi ]     # valence part
    dc0c = dm0m[:, param.nv_fi:, param.nv_fi: ]     # conduction part

    gauge_v = get_gauge(dv0v)
    gauge_c = get_gauge(dc0c)

    gauge_v_rev = np.flip(gauge_v,(1,2))

    np.savez("gaugemat.npz", gauge_v = gauge_v, gauge_v_rev=gauge_v_rev, \
             gauge_c = gauge_c, dm0m=dm0m)












