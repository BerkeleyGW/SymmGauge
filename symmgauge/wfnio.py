import h5py
import sys
import numpy as np


class Wfn:
    """This is a class for wavefucntion """

    def __init__(self, fname):

        self.fname = fname
        self.read_header()
        self.lines()

    def read_header(self):

        f_fi = h5py.File(self.fname, 'r')

        self.nk_fi = f_fi['/mf_header/kpoints/nrk'][()]
        self.rk = f_fi['/mf_header/kpoints/rk'][()]

        self.mnb_fi= f_fi ['/mf_header/kpoints/mnband'][()]
        # print("bands are {}".format(mnb_fi))

        self.ngk = f_fi['mf_header/kpoints/ngk'][()]
        ngk_sum = np.cumsum(self.ngk)
        self.ngk_sum = np.insert(ngk_sum, 0, 0, axis=0)     # dimension increase one
        self.ngkmax = np.max(self.ngk)
        self.ifmax = f_fi.get('mf_header/kpoints/ifmax')[()]
        self.ifmaxone = np.max(self.ifmax)

        ry2ev = 13.605662
        self.el = f_fi['/mf_header/kpoints/el'][()]*ry2ev

        self.bvec = f_fi['/mf_header/crystal/bvec'][()]


        f_fi.close()

    def lines(self):
        kpts = self.rk
        self.line1 = np.where(np.abs(kpts[:, 1])<0.00001)[0]
        self.line2 = np.where(np.abs(kpts[:, 1] - kpts[:, 0])<0.00001)[0]
        self.line3 = np.where(np.abs(kpts[:, 0])<0.00001)[0]


    def read_wfn(self, use_ngk=True):

        fname = self.fname
        f_fi = h5py.File(fname, 'r')

        nk_fi = f_fi['/mf_header/kpoints/nrk'][()]
        mnb_fi= f_fi ['/mf_header/kpoints/mnband'][()]
        print("Total bands are {}".format(mnb_fi))

        ngk = f_fi['mf_header/kpoints/ngk'][()]
        ngk_sum = np.cumsum(ngk)
        ngk_sum = np.insert(ngk_sum, 0, 0, axis=0)     # dimension increase one
        ngkmax = np.max(ngk)


        print("")
        print("Reading wfns ...")

        tmp = f_fi['wfns/coeffs'][()]
        tmp = tmp[:, :, :, 0] + 1j*tmp[:, :, :, 1]

        print("Finished Reading wfns ...")

        nb_tmp, nspinor, ncplx_tot = tmp.shape
        coeff_fi = np.zeros((nb_tmp, nspinor, nk_fi, ngkmax), dtype=complex)

        if use_ngk:
            for i, igk in enumerate(ngk):
                coeff_fi[:, :, i, :ngk[i]] = tmp[:, :, ngk_sum[i]:ngk_sum[i+1]]
        else:
            coeff_fi[:, :, :, :] = tmp.reshape(nb_tmp, nspinor, nk_fi, -1 )    # If we use hack version of BGW

        coeff_fi = coeff_fi.transpose(2, 0, 1, 3)     # nk x nb x nspinor x ngkmax
        # release memeory
        tmp = None
        f_fi.close()

        return coeff_fi


def read_wfn(fname, use_ngk=True):

    f_fi = h5py.File(fname, 'r')

    nk_fi = f_fi['/mf_header/kpoints/nrk'][()]
    mnb_fi= f_fi ['/mf_header/kpoints/mnband'][()]
    print("bands are {}".format(mnb_fi))

    ngk = f_fi['mf_header/kpoints/ngk'][()]
    ngk_sum = np.cumsum(ngk)
    ngk_sum = np.insert(ngk_sum, 0, 0, axis=0)     # dimension increase one
    ngkmax = np.max(ngk)


    print("")
    print("Reading wfns ...")

    tmp = f_fi['wfns/coeffs'][()]
    tmp = tmp[:, :, :, 0] + 1j*tmp[:, :, :, 1]

    print("Finished Reading wfns ...")

    nb_tmp, nspinor, ncplx_tot = tmp.shape
    coeff_fi = np.zeros((nb_tmp, nspinor, nk_fi, ngkmax), dtype=complex)

    if use_ngk:
        for i, igk in enumerate(ngk):
            coeff_fi[:, :, i, :ngk[i]] = tmp[:, :, ngk_sum[i]:ngk_sum[i+1]]
    else:
        coeff_fi[:, :, :, :] = tmp.reshape(nb_tmp, nspinor, nk_fi, -1 )    # If we use hack version of BGW

    coeff_fi = coeff_fi.transpose(2, 0, 1, 3)     # nk x nb x nspinor x ngkmax
    # release memeory
    tmp = None
    f_fi.close()

    return coeff_fi


def wfndot(v1, v2):
    # v1: nspinor x ngkmax
    # dot product for spinor wavefucntion

    if len(v1) == 2:
        result = v1[0, :].conj().dot(v2[0, :]) + v1[1, :].conj().dot(v2[1, :])
    elif len(v1) == 1:
        result = v1[0, :].conj().dot(v2[0, :])
    else:
        raise ValueError('dimension of wavefucntion is not correct')

    return result


def checksum(dvv):
    def msqure(a):
        return a.conj()*a
    a, b ,c = dvv.shape
    tmp = np.apply_along_axis(msqure, 1, dvv)
    return np.sum(tmp, axis=1)


def _overlap_2_center(fname1, fname2, nv_co, nc_co, nv_fi, nc_fi,\
                      idx_k0, gcut=None):

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

    coeff_co = wfnco.read_wfn()
    coeff_fi = wfnfi.read_wfn()

    if gcut:
        coeff_fi = coeff_fi[:, :, :, :gcut]
        coeff_co = coeff_co[:, :, :, :gcut]

    nbnd_co = nv_co + nc_co
    nbnd_fi = nv_fi + nc_fi
    nk_fi = coeff_fi.shape[0]
    dn0n = np.zeros((nk_fi, nbnd_co, nbnd_fi), dtype=complex)    # nk x co x fi

    print(" Start to calculate the overlap ")
    for k in range(nk_fi):
        for i, idx_co in enumerate(range(bstart_co, bend_co)):
            for j, idx_fi in enumerate(range(bstart_fi, bend_fi)):
                dn0n[k, i, j] = wfndot(coeff_co[idx_k0, idx_co], \
                                       coeff_fi[k, idx_fi])            # <phi|psi >

    print("Finish calculate the overlap")
    np.save("dn0n.npy", dn0n)
    np.savez("dn0n.npz", dn0n=dn0n, nv_co=nv_co, nv_fi=nv_fi)


if __name__ == "__main__":

    fname1 = 'wfn_co.h5'
    fname2 = 'wfn_co.h5'

    nv_co = 4
    nc_co = 5

    nv_fi = 2
    nc_fi = 2
    idx_k0 = 0


    _overlap_2_center(fname1, fname2, nv_co, nc_co, nv_fi, nc_fi, idx_k0)



   #  # check along certain direction
   #  wfn_fi = Wfn(fi_name)
   #  kpts = wfn_fi.rk
   #  line1 = np.where(np.abs(kpts[:, 1])<0.00001)[0]
   #  line2 = np.where(np.abs(kpts[:, 1] - kpts[:, 0])<0.00001)[0]
   #  dvv = dn0n[:, param.nv_co_k0-3:param.nv_co_k0 , 0:3 ]




















