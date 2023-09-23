import numpy as np
import h5py
from subprocess import call


class Xct:

    def __init__(self, fname='eigenvectors.h5',  vmtcls=None, gaugecls=None):


        self.init_from_file(fname)
        self.fname = fname

        if gaugecls:
            self.evecs_smth = self.acvk_gauge_trans(self.evecs, gaugecls)

            self.write_new_xct()

   #      if xy2lr:

   #          self.bIl, self.bIcp = self.bvecIlin_cir(self.bvec)
   #          self.dipoles_rl = self.rotate_dipole(self.dipoles, self.bIcp)

   #          if gaugecls:
   #              self.dipoles_rl_smth = self.gauge_trans(self.dipoles_rl, gaugecls)
   #              self.save_data()

        if vmtcls:
            print("calculate the r_eh")




    def init_from_file(self, fname):
       """

       """
       print('\n  Initializing exciton from file:', fname)

       with h5py.File(fname, 'r') as f:
           # !! Be careful that Q-shift is NOT exciton COM
           #self.Qpts = f.get('exciton_header/kpoints/exciton_Q_shifts')[()]
           self.Qpts = f.get('exciton_header/kpoints/Qpts')[()]
           self.kpts = f.get('exciton_header/kpoints/kpts')[()]
           self.nfk = f.get('exciton_header/kpoints/nk')[()]
           self.nevecs = f.get('exciton_header/params/nevecs')[()]

           self.nvb = f.get('exciton_header/params/nv')[()]
           self.ncb = f.get('exciton_header/params/nc')[()]
           self.xdim = f.get('exciton_header/params/bse_hamiltonian_size')[()]

           self.blat = f.get('mf_header/crystal/blat')[()]
           self.bvec = f.get('mf_header/crystal/bvec')[()]

           rkpts = f.get('mf_header/kpoints/rk')[()][:,:2]
           # there is another rk in exciton_header, which is within 1st BZ
           self.k_cart = np.dot(rkpts, self.bvec[0:2,0:2])       # FIX: to make it three dimension

           # eigenvalues in eV
           Ry2eV = 1/13.6057
           self.evals = f.get('exciton_data/eigenvalues')[()]/Ry2eV

           tmp = f['exciton_data/eigenvectors'][()]
           iis = 0; iQ = 0
           self.evecs = tmp[iQ,:,:,:,:,iis,0] + 1j*tmp[iQ,:,:,:,:,iis,1]


       return


    def acvk_gauge_trans(self, cvkfield, gaugecls):
        """
         X = acvk*|c><v| = acvk*<c_chi|c>* <v|v_chi> |c_chi><v_chi|
         gauge :  < chi | psi >,   nk * (nc + nv) * (nc+nv)
        """

        nX, nk, nc, nv = cvkfield.shape
        nv_fi = gaugecls.nv_fi
        nc_fi = gaugecls.nc_fi
        nk_fi = gaugecls.gauge_UN.shape[0]
        if nv < nv_fi or nc < nc_fi:
            raise Exception("The bands in vmtx files are not enough for gauge transformation")


        gauge_cc, gauge_vv = gaugecls.split_cc_vv(reorder=True)
        gauge_cc_p =np.array([mat.conj().T for mat in gauge_cc]) # <psi | chi>
        gauge_vv_p = np.array([mat.conj().T for mat in gauge_vv])

        from copy import deepcopy
        cvkfield_smth = deepcopy(cvkfield)
        cvkfield_trunc = cvkfield[:, :, :nc_fi, :nv_fi]
        cvkfield_trunc = np.einsum('kdc,skcv,kvw->skdw', gauge_cc, cvkfield_trunc, gauge_vv_p)

        cvkfield_smth[:,:,:nc_fi, :nv_fi] = cvkfield_trunc

        return cvkfield_smth


    def write_new_xct(self):

        new_avck = self.evecs_smth
        fname_in = self.fname

        fname_out = 'eigenvectors_smooth.h5'
        call(["cp", fname_in, fname_out])

        with h5py.File(fname_out, 'r+') as f:

            evecs = f.get('exciton_data/eigenvectors')[()]
            node = 'exciton_data/eigenvectors'
            if node in f:
                del f[node]

            print("Rewrite the matrix elements")
            shape_evecs = evecs.shape
            iis = 0; iQ = 0   # TODO: to generalize to nspin = 2 case
            f.create_dataset(node, shape_evecs, dtype='f8')
            f[node][iQ,:,:,:,:,iis,0] = np.real(new_avck)
            f[node][iQ,:,:,:,:,iis,1] = np.imag(new_avck)

        return
