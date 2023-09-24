import numpy as np
import h5py
from symmgauge.utility import svd_like

class Vmtxel:

    def __init__(self, vmtseedname='eigenvalues', inputmode='BGW', mffile='wfn.h5', \
                  gaugecls=None):
        """
        we need to switch from inputmode to be vmtfile
        """

        self.k_cart, self.bvec = self.read_mfs(mffile)    # currently focus on xy plane

        if (inputmode == 'BGW_ascii') or ("eigenvalues" in vmtseedname):
            self.vmtseedname= vmtseedname
            self.dipoles, self.de = self.init_from_bgw_ascii(vmtseedname)

        if gaugecls:
            self.dipoles_smth = self.gauge_trans(self.dipoles, gaugecls)

        self.bIl, self.bIcp = self.bvecIlin_cir(self.bvec)
        self.dipoles_rl = self.rotate_dipole(self.dipoles, self.bIcp)

        if gaugecls:
            self.dipoles_rl_smth = self.gauge_trans(self.dipoles_rl, gaugecls)
            self.save_data()


    def read_mfs(self, fname):
        if 'wfn' in fname or 'vector' in fname:
            return self._read_mfs_wfnORevec(fname)

    def _read_mfs_wfnORevec(self, fname):
        with h5py.File(fname, 'r') as f:
            if2D = True
            bvec = f.get('mf_header/crystal/bvec')[()][0:2,0:2]
            rkpts = f.get('mf_header/kpoints/rk')[()][:,:2]

        k_cart = np.dot(rkpts, bvec)

        return k_cart, bvec


    def bvecIlin_cir(self, bvec):
        """
        bvector direction onverted to linear direction (x, y)
        and circular direction (l, r)

        bvec_0:
          < ex | e1 >  < ey | e1 >
          < ex | e2>  < ey | e2 >
        lIb:
          < ex | e1 >  < ex | e2 >
          < ey | e1>  < ey | e2 >
        bIl:
          <e1 | ex>  <e1 | ey>
          <e2 | ex>  <e2 | ey>
        Dipole transformation:
                                        aim  , ain
            (d_m, d_n)  = (d_i, d_j) *(            )
                                        ajm  , ajn
        or using einsum rule,
            np.einsum("kcvi,im->kcvm",dipole_group, bIl)

        lIcp:
          <ex | e+>  <ex | e->
          <ey | e+>  <ey | e->

        """

        bvec_0 = np.zeros_like(bvec)
        bvec_0[0] = bvec[0]/np.linalg.norm(bvec[0])
        bvec_0[1] = bvec[1]/np.linalg.norm(bvec[1])

        lIb = bvec_0.transpose()
        bIl = np.linalg.inv(lIb) # TODO

        lIcp = np.sqrt(2)/2.* np.array([[1., 1.0],
                                        [1.j, -1.j]])               # TODO: + is above

        bIcp = np.dot(bIl, lIcp)

        return bIl, bIcp


    def init_from_bgw_ascii(self, vmtseedname):
        """
        The format of dipole:  <c|r|v>

        """

        # f1 = 'eigenvalues_b1_noeh.dat'
        # f2 = 'eigenvalues_b2_noeh.dat'
        # f3 = 'eigenvalues_b3_noeh.dat'

        f1 = vmtseedname + '_b1_noeh.dat'
        f2 = vmtseedname + '_b2_noeh.dat'
        f3 = vmtseedname + '_b3_noeh.dat'

        with open(f1) as f:
            lines = f.readlines()
            info = lines[2].split()
            ns = int(info[-5])
            nk = int(info[-3])
            nc = int(info[-2])
            nv = int(info[-1])


        dipoles = np.zeros((ns, nk, nc, nv, 3), dtype='complex128')
        dipoles_bgw = np.zeros((ns, nk, nc, nv, 3), dtype='complex128')
        de = np.zeros((ns, nk, nc, nv), dtype='float64')

        for i, f in enumerate([f1,f2,f3]):

            print("Loading files {}".format(i))
            data = np.loadtxt(f)

            dipole = (data[:,-2] + 1j*data[:,-1])

            newdata = dipole.reshape(nk,nc,nv,ns)
            newdata = newdata.transpose(3,0,1,2)
            dipoles_bgw[..., i] = newdata


            if i == 0:
                tmpev = data[:, 6]
                tmpev = tmpev.reshape(nk, nc, nv, ns)
                tmpev = tmpev.transpose(3,0,1,2)
                de[...] = tmpev

        dipoles = dipoles_bgw*(1j)   # TODO: to check the 1j

        # self.momentums = self.cal_momentum(dipoles_bgw, de)

        return dipoles, de



    def cal_momentum(self, dipoles_bgw, de):

        # TODO: we should check the imginary i
        Ry2eV = 13.605
        eV2Ry = 1 / Ry2eV

        de = de*eV2Ry
        ns, nk, nc, nv, npol = dipoles_bgw.shape

        momentums = np.zeros_like(dipoles_bgw)

        for i in range(ns):
            for j in range(nk):
                for k in range(nc):
                    for l in range(nv):
                        momentums[i,j,k,l,:] = de[i,j,k,l]*dipoles_bgw[i,j,k,l]
                        # TODO: if imginary i here?

        m_Ry = 0.5  # electron mass in Ry unit
        momentums = m_Ry*momentums

        # test = np.einsum("skcvp,skcv->skcvp", dipoles_bgw, de)
        return momentums

    def rotate_dipole(self, dipoles, ijImn):
        """
        input: dipoles: ns x nk x nc x nv x npol
        new scheme:

                                    aim  , ain
        (d_m, d_n)  = (d_i, d_j) *(            )
                                    ajm  , ajn

        """


        dip_sub = dipoles[...,:2]

        dipole_m = dip_sub[...,0] * ijImn[0,0] + dip_sub[...,1] * ijImn[1,0]
        dipole_n = dip_sub[...,0] * ijImn[0,1] + dip_sub[...,1] * ijImn[1,1]

        dipoles_new = np.zeros_like(dipoles)
        dipoles_new[...,0] = dipole_m
        dipoles_new[...,1] = dipole_n
        dipoles_new[...,2] = dipoles[...,2]

        # In the future:
        # np.einsum("kcvi,im->kcvm",dipole_group, ijImn)

        return dipoles_new

    def gauge_trans(self, dipoles, gaugecls):
        """
         <c_chi |r|v_chi > = <c_chi|c> <c|r|v>  <v|v_chi>
        """

        ns, nk, nc, nv, npol = dipoles.shape
        nv_fi = gaugecls.nv_fi
        nc_fi = gaugecls.nc_fi
        nk_fi = gaugecls.gauge_UN.shape[0]
        if nv < nv_fi or nc < nc_fi:
            raise Exception("The bands in vmtx files are not enough for gauge transformation")


        gauge_cc, gauge_vv = gaugecls.split_cc_vv(reorder=True)
        gauge_cc_p =np.array([mat.conj().T for mat in gauge_cc]) # <psi | chi>
        gauge_vv_p = np.array([mat.conj().T for mat in gauge_vv])

        from copy import deepcopy
        dipoles_smth = deepcopy(dipoles)
        dipoles_trunc = dipoles[:, :, :nc_fi, :nv_fi, :]
        dipoles_trunc = np.einsum('kdc,skcvp,kvw->skdwp', gauge_cc, dipoles_trunc, gauge_vv_p)

        dipoles_smth[:,:,:nc_fi, :nv_fi, :] = dipoles_trunc

        return dipoles_smth

    def save_data(self):

        try:
            self.dipoles_rl_smth
            np.savez("dipoles_gauged.npz", k_cart = self.k_cart, dipoles=self.dipoles, dipoles_rl_smth=self.dipoles_rl_smth, \
                     dipoles_rl=self.dipoles_rl)
        except AttributeError:
            print("dipoles_rl_smth not exit, only save the raw data")
            np.savez("dipoles_gauged.npz", k_cart = self.k_cart, dipoles=self.dipoles)


    def cal_k0_trans(self, idx_k0, cinds=None, vinds=None):
    # def cal_k0_trans(self, idx_k0, crange, vrange):

        iis = 0;
        inplane_inds = [0,1]    # assume in-plane dengenerate
        subrange = np.ix_(cinds, vinds, inplane_inds)
        tmp = self.dipoles_rl[iis, idx_k0][subrange]

        if len(cinds) == 2:
            print("\n Calculating transformation matrix in k0")
            print("   Conduction states are DOUBLY degenerate")
            print("   Use only one non-degnerate valence state that is optically coupled")
            print("   to the degenerate conduction bands")

            iv = 0
            tmp = tmp[:,iv,:]              # nc x npol,  <c|r|v>, only pick one v band
            submat = tmp.conj().T          # npol x nc, <v|r|c>, making pol to be axis 0
            k0_trans = svd_like(submat)    # return < c_reg | c_dft >

        elif len(vinds) == 2:    # v bands degenerate
            print("\n Calculating transformation matrix in k0")
            print("   Valence states are DOUBLY degenerate")
            print("   Use only one non-degnerate couduction state that is optically coupled")
            print("   to the degenerate valence bands")

            ic = 0
            tmp = tmp[ic,:,:]              # nv x npol,  <c|r|v>, only pick one v band
            submat = tmp.T                 # NOTE: no conjecture, but sitll making pol to be axis 0
            k0_trans = svd_like(submat)    # return < v_reg | v_dft >


        return k0_trans





