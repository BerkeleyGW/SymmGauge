import h5py
import sys
import numpy as np
from subprocess import call


def rotate_avck(cvkfield, gauge, nv, nc, reorder=True):

    # X = acvk*|c><v| = acvk*<c_chi|c>* <v|v_chi> |c_chi><v_chi|
    # gauge :  < chi | psi >,   nk * (nc + nv) * (nc+nv)

    gauge_cc = gauge[:, nv:, nv:]
    gauge_cc_p =np.array([mat.conj() for mat in gauge_cc]) # <psi | chi>

    gauge_vv = gauge[:, :nv, :nv]
    if reorder:
        gauge_vv = np.flip(gauge_vv, (1,2))    # To match the order of BGW
    gauge_vv_p = np.array([mat.conj() for mat in gauge_vv])

    if cvkfield.ndim == 4:    # acvk field
        nx, nk, nc_x, nv_x = cvkfield.shape
        # check:
        if nv != nv_x or nc != nc_x:
            raise Exception('Inconsist with the dimension of field and input nc and nv ')

        cvk_new = np.einsum('kdc,xkcv->xkdv', gauge_cc, cvkfield)
        cvk_new = np.einsum('xkdv,kvw ->xkdw', cvk_new, gauge_vv_p )

    return cvk_new


def write_new_xct(fname_in, fname_out, gauge, nv, nc):

    # fin = h5py.File(fname_in, "r")
    call(["cp", fname_in, fname_out])

    with h5py.File(fname_out, 'r+') as f:

        evecs = f.get('exciton_data/eigenvectors')[()]

        iis = 0; iQ = 0   # TODO: to generalize to nspin = 2 case
        xall = evecs[iis, :, :, :, :, iQ, 0] + \
            1j*evecs[iis, :, :, :, :, iQ, 1]         # nX x nk x nc x nv
        new_avck = rotate_avck(xall, gauge, nv, nc)

        node = 'exciton_data/eigenvectors'
        if node in f:
            del f[node]

        print("Rewrite the matrix elements")
        ns = 1; nQ = 1
        shape_evecs = evecs.shape
        f.create_dataset(node, shape_evecs, dtype='f8')
        f[node][iis,:,:,:,:,iQ,0] = np.real(new_avck)
        f[node][iis,:,:,:,:,iQ,1] = np.imag(new_avck)


    return



