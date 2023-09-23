import numpy as np
import h5py
import sys
import pickle
from copy import deepcopy
# from similarity import AbstractG6_2, similarity, rep_G6_2
from scipy.linalg import sqrtm
from symmgauge.wfnio import wfndot, Wfn


class GaugeKp:
    """
    This is a class for construction of smoth gauge in the script of kp theory
    """

    def __init__(self, wfn, nv_fi, nc_fi, idx_k0, gcut=None):

        self.wfn = wfn
        self.nv_fi = nv_fi
        self.nc_fi = nc_fi
        self.gcut = gcut
        self.idx_k0 = idx_k0

        self.dn0n = self.overlap2center(wfn, nv_fi, nc_fi, idx_k0)  # <u_dft_0|psi> , will be
                                                                    # updated to <u_reg_0 |psi >
        self.dn0chi = deepcopy(self.dn0n)
        self.dn0n_init = deepcopy(self.dn0n) # <u_dft_0|psi>

        self.gauge_UN = np.zeros_like(self.dn0n)
        for i, ele in enumerate(self.gauge_UN):
            self.gauge_UN[i] = np.eye(len(ele), dtype=complex)



    def overlap2center(self, wfn, nv_fi, nc_fi, idx_k0, gcut=None):

        ifmaxone_fi = wfn.ifmaxone
        print("Fermi level of fine-grid wfc: {}".format(ifmaxone_fi))
        bstart_fi = ifmaxone_fi - nv_fi
        bend_fi = ifmaxone_fi + nc_fi

        coeff_fi = wfn.read_wfn_range(bstart_fi, bend_fi, use_ngk=True)
        coeff_co_k0 = coeff_fi[idx_k0, ...]  # nb x nspin x ngvec   # here co is meaningless

        nbnd_fi = nv_fi + nc_fi
        nk_fi = coeff_fi.shape[0]
        dn0n = np.zeros((nk_fi, nbnd_fi, nbnd_fi), dtype=complex)    # nk x co x fi

        print(" Start to calculate the overlap ")
        print("The center kpoint is:")
        print(wfn.rk[idx_k0])

        for k in range(nk_fi):
            for i, idx_co in enumerate(range(bstart_fi, bend_fi)):
                for j, idx_fi in enumerate(range(bstart_fi, bend_fi)):
                    dn0n[k, i, j] = wfndot(coeff_co_k0[i], \
                                           coeff_fi[k, j])            # <phi|psi >

        return dn0n


    def update_gauge(self, bstart, bend, dn0n):

        """
        update gauge matrix.
        """
        # bstart = self.nv_fi + ibstart
        # bend = self.nv_fi + ibend
        band_range = list(range(bstart, bend))
        nk, nb, _ = dn0n.shape
        if bend > nb or bstart < 0 :
            raise Exception('setcted band is not in the bands provided')

        dm0m = dn0n[np.ix_(range(nk), band_range, band_range)]   # subspace

        gauge_sub = self.svd_like(dm0m)   # < chi| psi >

        # update the gauge field
        self.gauge_UN[np.ix_(range(nk), band_range, band_range)] = gauge_sub

        return self.gauge_UN

    def update_dn0n(self, bstart, bend, k0_trans=None):
        """
        update dn0n, making n0 with regular character, e.g. eigensate of rotation opeartor.
        NOT USES. k0_trans:  < dft_0 | reg_0 >
        k0_trans:  < reg_0 | psi_0 >
        """
        if k0_trans is not None:
            if bend - bstart != len(k0_trans):
                raise Exception('provided matrix dimension is not consistent with # selected bands')

            dn0n_old = self.dn0n
            band_range = list(range(bstart, bend))
            nk, nb, _ = dn0n_old.shape
            if bend > nb or bstart < 0 :
                raise Exception('setcted band is not in the bands provided')

            dm0m = dn0n_old[np.ix_(range(nk), band_range, band_range)]   # subspace
            # <reg_co_zero | fi >
            dm0m = np.einsum("ij,kjf->kif", k0_trans, dm0m)    # <reg_0|dft_0> * <dft_0 | fi >

            print("dn0n is updated")
            self.dn0n[np.ix_(range(nk), band_range, band_range)] = dm0m

            return self.dn0n

        else:
            return self.dn0n



    def svd_like(self, dm0m):
        '''get the gauge transformation matrix  '''
        dm0m_squre = [dm0m[i].dot(dm0m[i].conj().T) for i in range(len(dm0m))]
        dm0m_squre = np.array(dm0m_squre)
        dm0_chiv = np.array([sqrtm(m) for m in dm0m_squre])  # < phi | chi >  # squre root of matrix

        gauge_tran = np.array([np.linalg.inv(v).dot(u) for u, v in \
                               zip(dm0m, dm0_chiv) ])  # inv(<phi|chi>).<phi|psi> = <chi|psi>

        return gauge_tran

    def split_cc_vv(self, reorder=True):

        nv = self.nv_fi
        gauge_cc = self.gauge_UN[:, nv:, nv:]
        gauge_cc_p =np.array([mat.conj() for mat in gauge_cc]) # <psi | chi>

        gauge_vv = self.gauge_UN[:, :nv, :nv]
        if reorder:
            gauge_vv = np.flip(gauge_vv, (1,2))    # To match the order of BGW
        gauge_vv_p = np.array([mat.conj() for mat in gauge_vv])


        return gauge_cc, gauge_vv



# ================== following is not maintanted ==============

def _overlap_2_center(fname1, fname2, nv_co, nc_co, nv_fi, nc_fi,\
                      idx_k0, gcut=None):
    """
    need wfndot, Wfn
    """

    wfnco = Wfn(fname1)
    wfnfi = Wfn(fname2)

    ifmaxone = wfnco.ifmaxone
    ifmaxone_fi = wfnfi.ifmaxone
    print("Fermi level of coarse-grid wfc: {}".format(ifmaxone))
    print("Fermi level of fine-grid wfc: {}".format(ifmaxone_fi))
    bstart_co = ifmaxone - nv_co
    bend_co = ifmaxone + nc_co

    bstart_fi = ifmaxone_fi - nv_fi
    bend_fi = ifmaxone_fi + nc_fi

    coeff_fi = wfnfi.read_wfn()     # nk x nb x nspin x ngvec
    if fname1 == fname2:
        print("Input files are the same. Copy to coeff...")
        # coeff_co = coeff_fi
        coeff_co_k0 = coeff_fi[idx_k0, ...]  # nb x nspin x ngvec
    else:
        coeff_co_k0 = wfnco.read_wfn()
        coeff_co_k0 = coeff_co_k0[idx_k0,...]  # nb x nspin x ngvec

    if gcut:
        coeff_fi = coeff_fi[:, :, :, :gcut]
        coeff_co_k0 = coeff_co_k0[:, :, :gcut]

    nbnd_co = nv_co + nc_co
    nbnd_fi = nv_fi + nc_fi
    nk_fi = coeff_fi.shape[0]
    dn0n = np.zeros((nk_fi, nbnd_co, nbnd_fi), dtype=complex)    # nk x co x fi

    print(" Start to calculate the overlap ")
    print("The center kpoint is:")
    print(wfnco.rk[idx_k0])

    for k in range(nk_fi):
        for i, idx_co in enumerate(range(bstart_co, bend_co)):
            for j, idx_fi in enumerate(range(bstart_fi, bend_fi)):
                dn0n[k, i, j] = wfndot(coeff_co_k0[idx_co], \
                                       coeff_fi[k, idx_fi])            # <phi|psi >

    print("Finish calculate the overlap")
    np.save("dn0n.npy", dn0n)
    np.savez("dn0n.npz", dn0n=dn0n, nv_co=nv_co, nv_fi=nv_fi)
