import numpy as np
import h5py
from subprocess import call
from symmgauge.utility import svd_like


class Xct:

    def __init__(self, fname='eigenvectors.h5',  vmtcls=None, gaugecls=None,
                 deg_list=None):


        self.fname = fname
        self.init_from_file(fname)    # get self.evecs and others

        self.evecs_smth = None
        if gaugecls is not None:
            self.evecs_smth = self.acvk_gauge_trans(self.evecs, gaugecls)

            self.write_new_xct(self.evecs_smth, "eigenvectors_smooth.h5")


        if vmtcls is not None:
            print("Calculate r_eh, in both linear and circular coordinates")
            self.reh = self.cal_oeh(vmtcls.dipoles)
            self.reh_rl = self.cal_oeh(vmtcls.dipoles_rl)

            if deg_list is not None:
                from copy import deepcopy
                self.evecs_reg = deepcopy(self.evecs)
                for degpair in deg_list:
                    reg_trans = self.gen_reg_trans(degpair, self.reh_rl)

                    # update self.evecs_reg
                    self.regularized_acvk(self.evecs_reg, degpair, reg_trans)

                print("Finished regularized selected degnerate exciton pairs")

                if gaugecls is not None:
                    self.evecs_reg_smth = self.acvk_gauge_trans(self.evecs_reg, gaugecls)

                    self.write_new_xct(self.evecs_reg_smth, "eigenvectors_reg_smooth.h5")




    def init_from_file(self, fname):
        """

        """
        print('\n  Initializing exciton from file:', fname)

        with h5py.File(fname, 'r') as f:
            # !! Be careful that Q-shift is NOT exciton COM
            #self.Qpts = f.get('exciton_header/kpoints/exciton_Q_shifts')[()]
            #self.Qpts = f.get('exciton_header/kpoints/Qpts')[()]
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


    def cal_oeh(self,vmtx_noeh):

        """
        oeh is defined as:
        <S|o|0> = conj(A_cvk)* <c|o|v>

        """

        ns, nk, nc, nv, npol = vmtx_noeh.shape
        if nc < self.ncb or nv < self.nvb:
            raise Exception("The bands in vmtx files are not enough")
        inds = np.ix_(range(ns),range(nk),range(self.ncb),\
                                    range(self.nvb),range(3))

        iss = 0
        oeh = np.einsum('xkcv,kcvn->xn', self.evecs.conj(),\
                                               vmtx_noeh[inds][iss])

        return oeh

    def gen_reg_trans(self, degpair, oeh):
        """
        input matxls are:   <S|o|0>
        oeh: shape, nS x npol
        """
        if len(degpair) != 2:
            raise Exception('degpair should only contain two elements')
        inplane_pol = [0,1]
        inds = np.ix_(degpair, inplane_pol)
        tmp = oeh[inds]     # 2 x 2 matrix

        conjtmp = tmp.conj().T      # <S|o|0> ----> <0|o^deg |S>, and pol is axis 0

        reg_trans = svd_like(conjtmp)     # <S_reg | S_bgw>

        return reg_trans

    def regularized_acvk(self, acvk, degpair, reg_trans):
        """
        acvk =  |v><c| X >  ---->   acvk_reg = |v><c| X><X|X_reg >
        reg_trans: 2 x 2 matrix,  <S_reg| S_bgw >
        """
        if len(reg_trans) != 2:
            raise Exception('reg_trans should be a 2x2 matrix')
        sub_acvk = acvk[degpair, :, :, :]

        sub_acvk_reg = np.einsum("xcvk,xy->ycvk", sub_acvk, reg_trans.conj().T)

        acvk[degpair, :, :, :] = sub_acvk_reg

        return

    def acvk_gauge_trans(self, cvkfield, gaugecls):
        """
         X = acvk*|c><v| = acvk*<c_chi|c>* <v|v_chi> |c_chi><v_chi|
           = bcvk |c_chi><v_chi|
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


    def write_new_xct(self, new_avck, fname_out='eigenvectors_smooth.h5'):

        fname_in = self.fname

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
