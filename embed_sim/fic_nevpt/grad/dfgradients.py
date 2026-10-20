import ctypes
from pyscf import scf, fci, df, lib
from pyscf.cc import ccsd, _ccsd
from pyscf.ao2mo import _ao2mo
from pyscf.df.grad import mcpdft
from pyscf.lib import logger
from pyscf.fci import cistring
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.mcscf.addons import StateAverageMixFCISolver
from pyscf.ao2mo.outcore import balance_partition
from pyscf.df.grad.rhf import _int3c_wrapper, _gen_metric_solver
from pyscf.grad import casscf as casscf_grad
from pyscf.df.grad import casscf as dfcasscf_grad
import numpy as np
from numpy import einsum
from functools import partial
from functools import reduce
einsum = partial(einsum, optimize=True)

CLASSES = ("0", "+1", "-1", "+2", "-2", "0'", "+1'", "-1'")

from .derivatives import (compute_t1_0, compute_t1_p1, compute_t1_m1, compute_t1_p2, compute_t1_m2, compute_t1_0p, compute_t1_p1p, compute_t1_m1p)
from .semicanonical import get_semicanonical_response
from .gradients import Gradients
from ..rdm import make_tran_rdm1234

def _class_kernel(nevpt_grad, CLASS):
    _class_kernels = {
        "0"  : compute_t1_0,
        "+1" : compute_t1_p1,
        "-1" : compute_t1_m1,
        "+2" : compute_t1_p2,
        "-2" : compute_t1_m2,
        "0'" : compute_t1_0p,
        "+1'": compute_t1_p1p,
        "-1'": compute_t1_m1p
        }
    return _class_kernels[CLASS](nevpt_grad)

def grad(nevpt_grad, atmlst, pre_kernel=False):
    nevpt = nevpt_grad.nevpt
    for CLASS in nevpt_grad.classes:
        _class_kernel(nevpt_grad, CLASS)
    c = nevpt_grad.c
    a = nevpt_grad.a
    e = nevpt_grad.e
    for name, key in {'aa':(a,a), 'ca':(c,a), 'ce':(c,e), 'ae':(a,e)}.items():
        value = nevpt_grad.h1eff.__dict__[name]
        nevpt_grad.rdm1[key] += value
    nevpt_grad.rdm1 = 0.5 * (nevpt_grad.rdm1 + nevpt_grad.rdm1.T)
    d2 = tuple((nevpt_grad.v2e.__dict__[dkey] for dkey in nevpt_grad.v2e._keys))
    _rdm2_mo2ao(nevpt_grad.nevpt, d2, nevpt_grad.nevpt.mo, nevpt_grad.v2e.fdm2)

    max_memory = max(0, nevpt_grad.max_memory - lib.current_memory()[0])
    g_dm2, Imat_dm2 = _contract_dfdm2(nevpt_grad, nevpt_grad.v2e.fdm2, atmlst, max_memory)
    nevpt_grad._g_dm2 = g_dm2
    nevpt_grad._Imat_dm2 = Imat_dm2

    nevpt_grad.mo_energy[c] = nevpt_grad.mo_energy.c
    nevpt_grad.mo_energy[e] = nevpt_grad.mo_energy.e

    if pre_kernel:
        return nevpt_grad
    else:
        converged, Lvec = nevpt_grad.solve_lagrange(mo=nevpt.mo, ci=nevpt.ci, eris=nevpt.eris)[:2]
        if not converged:
            nevpt.log.warn('Z-vector equation did not converge!')

        mc_grad = nevpt_grad.mc.Gradients()
        mc_grad.auxbasis_response = nevpt_grad.auxbasis_response
        g = nevpt_grad.get_grad_skeleton(atmlst=atmlst)
        with lib.temporary_env(casscf_grad, Gradients=dfcasscf_grad.Gradients):
            g += mc_grad.get_ham_response(state=nevpt_grad.state, atmlst=atmlst, verbose=nevpt_grad.verbose,
                                          mo=nevpt_grad.nevpt.mo, ci=nevpt_grad.nevpt.ci, eris=nevpt_grad.eris, mf_grad=None)
        g += mc_grad.get_LdotJnuc(Lvec, state=nevpt_grad.state, atmlst=atmlst, verbose=nevpt_grad.verbose,
                                  mo=nevpt_grad.nevpt.mo, ci=nevpt_grad.nevpt.ci, eris=nevpt_grad.eris, mf_grad=None)
        return g

class DFGradients(Gradients):
    _keys = ['verbose', 'mol', 'nevpt', 'base', 'max_memory', 'mc', 'with_df', 'canonstep', 'auxbasis_response', 'unit', 'atmlst', 'de', 'classes', 'excluded_classes', 'state', 'e_mcscf', 'weights', 'nroots', 'ngorb', 'spin_states', 'na_states', 'nb_states', 'nci', 'nlag', 'level_shift', 'conv_atol', 'conv_rtol', 'max_cycle', '_t1', 'c', 'a', 'e', 'mo_energy', 'h1eff', 'v2e', 'rdms', 'rdm1', 'rdm2', 'rdm1e']

    _keys = _keys + ['ci_response_backend', 'ci_response_stats']

    def __init__(self, nevpt):
        self.with_df = getattr(nevpt, 'with_df', None)
        if self.with_df is None:
            raise ValueError
        self.verbose = nevpt.verbose
        self.mol = nevpt.mol
        self.nevpt = nevpt
        self.base = self.nevpt.base
        self.max_memory = self.nevpt.max_memory
        self.mc = self.nevpt.mc
        self.canonstep = self.nevpt.canonstep
        self.auxbasis_response = True
        self.unit = 'au'
        self.atmlst = None
        self.de = None
        self.classes = self.nevpt.classes
        self.excluded_classes = self.nevpt.excluded_classes
        self.state = self.nevpt.state
        self.e_mcscf = self.mc.e_states
        self.weights = self.mc.weights
        self.nroots = self.mc.fcisolver.nroots
        self.ngorb = np.count_nonzero(self.mc.uniq_var_indices(self.nevpt.nmo, self.mc.ncore, self.mc.ncas, self.mc.frozen))
        self.nroots = self.mc.fcisolver.nroots
        neleca, nelecb = _unpack_nelec(self.mc.nelecas)
        self.spin_states = [neleca - nelecb,] * self.nroots
        self.na_states = [cistring.num_strings(self.mc.ncas, neleca),] * self.nroots
        self.nb_states = [cistring.num_strings(self.mc.ncas, nelecb),] * self.nroots
        if isinstance(self.mc.fcisolver, StateAverageMixFCISolver):
            self.nroots = p0 = 0
            for solver in self.mc.fcisolver.fcisolvers:
                self.nroots += solver.nroots
                nea, neb = self.mc.fcisolver._get_nelec(solver, (neleca, nelecb))
                nstr_a = cistring.num_strings(self.mc.ncas, nea)
                nstr_b = cistring.num_strings(self.mc.ncas, neb)
                for p1 in range (p0, self.nroots):
                    self.spin_states[p1] = nea - neb
                    self.na_states[p1] = nstr_a
                    self.nb_states[p1] = nstr_b
                p0 = self.nroots
        self.nci = sum ([na * nb for na, nb in zip(self.na_states, self.nb_states)])
        self.nlag = self.ngorb + self.nci

        self.level_shift = 1e-8
        self.conv_atol = 1e-12
        self.conv_rtol = 1e-7
        self.max_cycle = 50
        self._hessian_gauge(ci=self.nevpt.ci, state=self.state)

        self._t1 = self.nevpt._t1_
        self.c = slice(0, nevpt.ncore)
        self.a = slice(nevpt.ncore, nevpt.nocc)
        self.e = slice(nevpt.nocc, nevpt.nmo)
        self.mo_energy = lib.tag_array(
            np.zeros_like(self.nevpt.mo_energy),
            c = np.zeros_like(self.nevpt.mo_energy[self.c]),
            a = np.zeros_like(self.nevpt.mo_energy[self.a]),
            e = np.zeros_like(self.nevpt.mo_energy[self.e])
        )
        self.h1eff = lib.tag_array(
            np.zeros_like(self.nevpt.h1eff),
            aa = np.zeros_like(self.nevpt.h1eff[self.a,self.a]),
            ca = np.zeros_like(self.nevpt.h1eff[self.c,self.a]),
            ce = np.zeros_like(self.nevpt.h1eff[self.c,self.e]),
            ae = np.zeros_like(self.nevpt.h1eff[self.a,self.e])
        )
        self.v2e = _ERIS(self.nevpt)
        self.rdms = _RDMS(self.nevpt)
        self.rdm1 = np.zeros([self.nevpt.nmo]*2)
        self.rdm1e = np.zeros([self.nevpt.nmo]*2)
        self.eris = self.nevpt.eris

    def get_ci_response_legacy(self):
        """Original CI-unit-vector / transition-RDM implementation, retained verbatim."""
        nevpt = self.nevpt
        state_ci = nevpt.state_ci
        gci = np.zeros_like(state_ci)
        state_ci0 = np.zeros_like(state_ci)
        for idx in range(state_ci.size):
            state_ci0.flat[idx] = 1.0
            tran_rdms = _RDMS(nevpt)
            tran_rdm1234 = make_tran_rdm1234(state_ci0, state_ci, nevpt.ncas, nevpt.nelecas)
            for i, name in enumerate(['ca', 'ccaa', 'cccaaa', 'ccccaaaa']):
                tran_rdms.__dict__[name] += tran_rdm1234[i]
            # tran_rdm1234 = make_tran_rdm1234(state_ci, state_ci0, nevpt.ncas, nevpt.nelecas)
            # for i, name in enumerate(['ca', 'ccaa', 'cccaaa', 'ccccaaaa']):
            #     tran_rdms.__dict__[name] += tran_rdm1234[i]
            tran_rdms.ca += tran_rdms.ca.transpose(1,0)
            tran_rdms.ccaa += tran_rdms.ccaa.transpose(2,3,0,1)
            tran_rdms.cccaaa += tran_rdms.cccaaa.transpose(3,4,5,0,1,2)
            if np.sum(nevpt.nelecas) > 1:
                tran_rdms.ccccaaaa += tran_rdms.ccccaaaa.transpose(4,5,6,7,0,1,2,3)
            gci.flat[idx] += einsum('Xx,Xx', self.rdms.ca, tran_rdms.ca)
            gci.flat[idx] += einsum('XYxy,XYxy', self.rdms.ccaa, tran_rdms.ccaa)
            gci.flat[idx] += einsum('XYZxyz,XYZxyz', self.rdms.cccaaa, tran_rdms.cccaaa)
            if np.sum(nevpt.nelecas) > 1:
                gci.flat[idx] += einsum('XYZWxyzw,XYZWxyzw', self.rdms.ccccaaaa, tran_rdms.ccccaaaa)
            state_ci0.flat[idx] = 0.0
        return gci

    def get_wfn_response(self, **kwargs):
        mol = self.mol
        nevpt = self.nevpt
        nmo = nevpt.nmo
        ncore = nevpt.ncore
        c = self.c
        a = self.a

        state_ci = nevpt.state_ci
        gci = self.get_ci_response()
    
        D = np.zeros((nmo,nmo))
        D[c,c] = 2 * np.eye(ncore)
        if self.canonstep == 0 or self.canonstep == 1:
            D[a,a] = nevpt.casdm1
        elif self.canonstep == 2:
            D[a,a] = nevpt.sa_casdm1
        else:
            raise NotImplementedError
        h = nevpt.h1e
        rdm1_ao = reduce(lib.dot, (nevpt.mo, self.rdm1, nevpt.mo.T))
        self.rdm1[range(nmo),range(nmo)] += self.mo_energy
        dm1e = np.diag(self.mo_energy)
        D_ao = reduce(lib.dot, (nevpt.mo, D, nevpt.mo.T))
        dm1e_ao = reduce(lib.dot, (nevpt.mo, dm1e, nevpt.mo.T))
        Dcore_ao = 2 * lib.dot(nevpt.mo.c, nevpt.mo.c.T)
        dm_list = np.array([rdm1_ao, Dcore_ao, D_ao, dm1e_ao])
        vj, vk = nevpt.mf.get_jk(dm=dm_list)
        veff = vj - 0.5 * vk
        veff_dm1e = einsum('ip,ij,jq->pq', nevpt.mo, veff[3], nevpt.mo)
        if self.canonstep == 0 or self.canonstep == 1:
            gci += 2.0 * fci.direct_spin1.contract_1e(veff_dm1e[a,a], state_ci, nevpt.ncas, nevpt.nelecas)
        elif self.canonstep == 2:
            gci_state = gci.copy()
            gci = [2.0 * fci.direct_spin1.contract_1e(veff_dm1e[a,a], ci, nevpt.ncas, nevpt.nelecas) / nevpt.nstates for ci in nevpt.ci]
            gci[nevpt.state] += gci_state
    
        gorb = lib.dot(h, self.rdm1.T) + lib.dot(h.T, self.rdm1)
        Imat = self._Imat_dm2.copy()
        Imat += 2 * lib.dot(veff[0], dm_list[1])
        Imat += 2 * lib.dot(veff[1], dm_list[0])
        Imat += 2 * lib.dot(veff[2], dm_list[3])
        Imat += 2 * lib.dot(veff[3], dm_list[2])
        gorb += reduce(lib.dot, (nevpt.mo.T, Imat, nevpt.mf.get_ovlp(), nevpt.mo))
        
        gorb_semi, gci_semi, rdm1_semi, X_semi_ao, D_semi_ao = get_semicanonical_response(self, gorb)
        self.rdm1 += rdm1_semi
        self._X_semi_ao = X_semi_ao
        self._D_semi_ao = D_semi_ao
    
        gorb = gorb + gorb_semi
        self.rdm1e = 0.5 * (gorb + gorb.T)

        Y = gorb - gorb.T
        y = [np.array(y_) for y_ in gci_semi]
        if self.canonstep == 0 or self.canonstep == 1:
            y[nevpt.state] += gci
        elif self.canonstep == 2:
            y = [np.array(gci_) + np.array(y_) for gci_, y_ in zip(gci, gci_semi)]
        for k, y_ in enumerate(y):
            y_ -= einsum('ij,ij', nevpt.ci[k], y_) * nevpt.ci[k]
        
        bvec = self.pack_uniq_var(Y, y)
        bvec += self.mc.Gradients().get_wfn_response(state=self.state, mo=nevpt.mo, ci=nevpt.ci)
        return self._project_bvec(bvec)

    def get_grad_skeleton(self, rdm1=None, rdm1e=None, atmlst=None):
        if rdm1 is None: rdm1 = self.rdm1
        if rdm1e is None: rdm1e = self.rdm1e
        if atmlst is None: atmlst = self.atmlst

        mol = self.mol
        nao = mol.nao
        nevpt = self.nevpt
        mo = nevpt.mo
        mf = nevpt.mf
        c = self.c
        a = self.a
        ncore = nevpt.ncore
        mf_grad = mf.Gradients()

        rdm1_ao = einsum('pq,ip,jq->ij', rdm1, mo, mo)
        rdm1e_ao = einsum('pq,ip,jq->ij', rdm1e, mo, mo)
        D_core_ao = 2 * lib.dot(nevpt.mo.c, nevpt.mo.c.T)
        D_ao = np.zeros((nao,nao))
        D_ao[c,c] = 2 * np.eye(ncore)
        if self.canonstep == 0 or self.canonstep == 1:
            D_ao[a,a] = nevpt.casdm1
        elif self.canonstep == 2:
            D_ao[a,a] = nevpt.sa_casdm1
        D_ao = einsum('pq,ip,jq->ij', D_ao, mo, mo)
        De_ao = np.diag(self.mo_energy)
        De_ao = einsum('pq,ip,jq->ij', De_ao, mo, mo)
        Deff_ao = rdm1_ao - De_ao - self._X_semi_ao
        dm_list = np.array([Deff_ao, D_core_ao, D_ao, De_ao, self._X_semi_ao, self._D_semi_ao])

        hcore_deriv = mf_grad.hcore_generator(mol)
        s1ao = mf_grad.get_ovlp(mol)
        vj1ao, vk1ao = mf_grad.get_jk(dm=dm_list)
        veff1ao = vj1ao - 0.5 * vk1ao
        vaux = vj1ao.aux - 0.5 * vk1ao.aux

        offsetdic = mol.offset_nr_by_atom()
        g = self._g_dm2.copy()
        for k, ia in enumerate(atmlst):
            _, _, p0, p1 = offsetdic[ia]
            h1ao = hcore_deriv(ia)
            g[k] += einsum('tij,ij->t', h1ao, rdm1_ao)
            g[k] -= einsum('tij,ij->t', s1ao[:, p0:p1], rdm1e_ao[p0:p1])
            g[k] += 2 * einsum('tij,ij->t', veff1ao[0][:, p0:p1], dm_list[1][p0:p1])
            g[k] += 2 * einsum('tij,ij->t', veff1ao[1][:, p0:p1], dm_list[0][p0:p1])
            g[k] += 2 * einsum('tij,ij->t', veff1ao[2][:, p0:p1], dm_list[3][p0:p1])
            g[k] += 2 * einsum('tij,ij->t', veff1ao[3][:, p0:p1], dm_list[2][p0:p1])
            g[k] += 2 * einsum('tij,ij->t', veff1ao[4][:, p0:p1], dm_list[5][p0:p1])
            g[k] += 2 * einsum('tij,ij->t', veff1ao[5][:, p0:p1], dm_list[4][p0:p1])

            if self.auxbasis_response:
                g[k] += (vaux[0,1][ia] + vaux[1,0][ia])
                g[k] += (vaux[2,3][ia] + vaux[3,2][ia])
                g[k] += (vaux[4,5][ia] + vaux[5,4][ia])
        return g

    def kernel(self, atmlst=None):
        cput0 = (logger.process_clock(), logger.perf_counter())

        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst
        if atmlst is None:
            self.atmlst = atmlst = list(range(self.mol.natm))

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        self.de = self.grad(atmlst=atmlst)

        logger.timer(self, 'FIC-NEVPT2 gradients', *cput0)
        self._finalize()
        return self.de

    def grad(self, atmlst):
        return grad(self, atmlst)

def _cp(a):
    return np.asarray(a, order='C')

def _rdm2_mo2ao(nevpt, d2, mo_coeff, fsave=None):

    log = logger.Logger(nevpt.stdout, nevpt.verbose)
    time1 = logger.process_clock(), logger.perf_counter()
    if fsave is None:
        incore = True
        fsave = lib.H5TmpFile()
    else:
        incore = False
    daaaa, daaae, daeae, dcaaa, dcaae, dcaca, dcace, dceaa, dceae, dcece = d2

    ncore, nextern, ncas = dceaa.shape[:3]
    nocc = ncore + ncas
    mo_coeff = np.asarray(mo_coeff, order='F')
    nao, nmo = mo_coeff.shape
    nao_pair = nao * (nao+1) // 2

    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s1
    fmm = _ccsd.libcc.CCmmm_transpose_sum
    pao_loc = ctypes.POINTER(ctypes.c_void_p)()
    def _trans(vin, orbs_slice, out=None):
        nrow = vin.shape[0]
        if out is None:
            out = np.empty((nrow,nao_pair))
        fdrv(ftrans, fmm,
             out.ctypes.data_as(ctypes.c_void_p),
             vin.ctypes.data_as(ctypes.c_void_p),
             mo_coeff.ctypes.data_as(ctypes.c_void_p),
             ctypes.c_int(nrow), ctypes.c_int(nao),
             (ctypes.c_int*4)(*orbs_slice), pao_loc, ctypes.c_int(0))
        return out

    fswap = lib.H5TmpFile()
    max_memory = nevpt.max_memory - lib.current_memory()[0]
    blksize = int(max_memory*1e6/8/(nao_pair+nmo**2))
    blksize = min(nao_pair, max(ccsd.BLKMIN, blksize))
    fswap.create_dataset('o', (nocc,ncas+nextern,nao_pair), 'f8', chunks=(nocc,min(nocc,ncas+nextern),blksize))
    
    for p0, p1 in lib.prange(0, ncas, nocc):
        buf1 = np.zeros((nocc,p1-p0,nocc,ncas+nextern))
        buf1[ncore:,:,ncore:,:ncas] = daaaa[:,p0:p1]
        buf1[ncore:,:,ncore:,ncas:] = daaae[:,p0:p1]
        buf1[:ncore,:,ncore:,:ncas] = dcaaa[:,p0:p1]
        buf1[:ncore,:,ncore:,ncas:] = dcaae[:,p0:p1]
        buf1[:ncore,:,:ncore,:ncas] = dcaca[:,p0:p1]
        buf1[:ncore,:,:ncore,ncas:] = dcace[:,p0:p1]
        buf1 = buf1.transpose(1,0,3,2).reshape((p1-p0)*nocc,-1)
        buf1 = _trans(buf1, (ncore,nmo,0,nocc))
        fswap['o'][:,p0:p1] = buf1.reshape(p1-p0,nocc,nao_pair).transpose(1,0,2)
    time1 = log.timer_debug1('_rdm2_mo2ao pass 1', *time1)
    buf1 = None
    
    for p0, p1 in lib.prange(0, nextern, nocc):
        buf1 = np.zeros((nocc,p1-p0,nocc,ncas+nextern))
        buf1[ncore:,:,ncore:,ncas:] = daeae[:,p0:p1]
        buf1[:ncore,:,ncore:,:ncas] = dceaa[:,p0:p1]
        buf1[:ncore,:,ncore:,ncas:] = dceae[:,p0:p1]
        buf1[:ncore,:,:ncore,ncas:] = dcece[:,p0:p1]
        buf1 = buf1.transpose(1,0,3,2).reshape((p1-p0)*nocc,-1)
        buf1 = _trans(buf1, (ncore,nmo,0,nocc))
        fswap['o'][:,ncas+p0:ncas+p1] = buf1.reshape(p1-p0,nocc,nao_pair).transpose(1,0,2)
    time1 = log.timer_debug1('_rdm2_mo2ao pass 2', *time1)
    buf1 = None

    gsave = fsave.create_dataset('dm2', (nao_pair,nao_pair), 'f8', chunks=(blksize,blksize))
    for p0, p1 in lib.prange(0, nao_pair, blksize):
        buf1 = _cp(fswap['o'][:,:,p0:p1].transpose(2,0,1).reshape(p1-p0,-1))
        buf2 = _trans(buf1, (0,nocc,ncore,nmo))
        if p0 > 0:
            buf1 = _cp(gsave[:p0,p0:p1])
            buf1[:p0,:p1-p0] += buf2[:p1-p0,:p0].T
            buf2[:p1-p0,:p0] = buf1[:p0,:p1-p0].T
            gsave[:p0,p0:p1] = buf1
        lib.transpose_sum(buf2[:,p0:p1], inplace=True)
        gsave[p0:p1] = buf2 * 0.5
    time1 = log.timer_debug1('_rdm2_mo2ao pass 3', *time1)
    if incore:
        return fsave['dm2'][:]
    else:
        return fsave

class _ERIS:
    _keys = ['aaaa', 'aaae', 'aeae', 'caaa', 'caae', 'caca', 'cace', 'ceaa', 'ceae', 'cece']

    def __init__(self, nevpt):
        ncore = nevpt.ncore
        ncas = nevpt.ncas
        nextern = nevpt.nextern

        fdm2 = lib.H5TmpFile()
        dtype = 'd'
        def create_dataset(key, shape, dtype, chunks):
            if 0 in shape:
                fdm2.create_dataset(key, shape, dtype)
            else:
                fdm2.create_dataset(key, shape, dtype, chunks=chunks)
        create_dataset('daaaa', (ncas,ncas,ncas,ncas), dtype, chunks=(ncas,ncas,1,ncas))
        create_dataset('daaae', (ncas,ncas,ncas,nextern), dtype, chunks=(ncas,ncas,1,nextern))
        create_dataset('daeae', (ncas,nextern,ncas,nextern), dtype, chunks=(ncas,nextern,1,nextern))
        create_dataset('dcaaa', (ncore,ncas,ncas,ncas), dtype, chunks=(ncore,ncas,1,ncas))
        create_dataset('dcaae', (ncore,ncas,ncas,nextern), dtype, chunks=(ncore,ncas,1,nextern))
        create_dataset('dcaca', (ncore,ncas,ncore,ncas), dtype, chunks=(ncore,ncas,1,ncas))
        create_dataset('dcace', (ncore,ncas,ncore,nextern), dtype, chunks=(ncore,ncas,1,nextern))
        create_dataset('dceaa', (ncore,nextern,ncas,ncas), dtype, chunks=(ncore,nextern,1,ncas))
        create_dataset('dceae', (ncore,nextern,ncas,nextern), dtype, chunks=(ncore,nextern,1,nextern))
        create_dataset('dcece', (ncore,nextern,ncore,nextern), dtype, chunks=(ncore,nextern,1,nextern))
        self.fdm2 = fdm2
        self.aaaa = self.fdm2['daaaa']
        self.aaae = self.fdm2['daaae']
        self.aeae = self.fdm2['daeae']
        self.caaa = self.fdm2['dcaaa']
        self.caae = self.fdm2['dcaae']
        self.caca = self.fdm2['dcaca']
        self.cace = self.fdm2['dcace']
        self.ceaa = self.fdm2['dceaa']
        self.ceae = self.fdm2['dceae']
        self.cece = self.fdm2['dcece']

class _RDMS:
    def __init__(self, nevpt):
        ncas = nevpt.ncas
        self.ca = np.zeros([ncas]*2)
        self.ccaa = np.zeros([ncas]*4)
        self.cccaaa = np.zeros([ncas]*6)
        self.ccccaaaa = np.zeros([ncas]*8)

# Copied from df.grad.ccsd in pyscf==2.15.0 to adapt older verisons
def _auxmol(nevpt):
    with_df = nevpt.with_df
    auxmol = with_df.auxmol
    if auxmol is None:
        auxmol = df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
    return auxmol

def _pair_weights(nao):
    # Undo the double counting of the off-diagonal elements of a symmetric
    # matrix stored in the lower-triangular packed layout.
    w = np.ones(nao*(nao+1)//2)
    diag = np.arange(nao)
    w[diag*(diag+1)//2 + diag] = .5
    return w

def solve_df_rdm2(nevpt_grad, fdm2, max_memory=None):
    '''Solve (P|Q) d_Quv = (P|kl) Gamma_uv,kl for the AO-basis DF 2-PDM.

    Args:
        cc_grad : DF-CCSD gradients method object
        fdm2 : HDF5 group holding the AO 2-PDM 'dm2' written by
            :func:`pyscf.grad.ccsd._rdm2_mo2ao`

    Returns:
        (int3c, dferi, dfdm2), each of shape (naux, nao_pair) in the
        lower-triangular AO pair layout:
            int3c[P,uv] = (P|uv)
            dferi[P,uv] = (P|Q)^-1 (Q|uv)
            dfdm2[P,uv] = (P|Q)^-1 (Q|kl) Gamma_uv,kl
    '''
    mol = nevpt_grad.mol
    auxmol = _auxmol(nevpt_grad.nevpt)
    nao, nbas = mol.nao, mol.nbas
    naux = auxmol.nao
    nao_pair = nao * (nao+1) // 2
    aux_loc = auxmol.ao_loc
    if max_memory is None:
        max_memory = nevpt_grad.max_memory

    int3c = np.empty((naux, nao_pair))
    get_int3c = _int3c_wrapper(mol, auxmol, 'int3c2e', 's2ij')
    blksize = max(1, min(int(max_memory*.2e6/8/max(nao_pair, 1)), naux))
    for shl0, shl1, nL in balance_partition(aux_loc, blksize):
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        int3c[p0:p1] = get_int3c((0, nbas, 0, nbas, shl0, shl1)).T

    solve_j2c = _gen_metric_solver(auxmol.intor('int2c2e', aosym='s1'))
    dferi = solve_j2c(int3c.copy())

    dm2 = fdm2['dm2']
    dfdm2 = np.empty((naux, nao_pair))
    dferi_w = dferi * _pair_weights(nao)
    blksize = max(1, min(int(max_memory*.2e6/8/max(naux, 1)), nao_pair))
    for p0, p1 in lib.prange(0, nao_pair, blksize):
        dfdm2[:,p0:p1] = lib.dot(dferi_w, np.asarray(dm2[p0:p1]).T)
    return int3c, dferi, dfdm2

def _contract_dfdm2(nevpt_grad, fdm2, atmlst, max_memory=None):
    '''DF counterpart of the four-index 2-PDM contraction in
    :func:`pyscf.grad.ccsd.grad_elec`.

    Returns:
        de : ndarray of shape (len(atmlst),3)
            d/dX of 0.5 sum_pqrs Gamma_pq,rs (pq|rs)_DF
        Imat : ndarray of shape (nao,nao)
            Imat[p,q] = sum_{i,kl} (ip|kl)_DF Gamma_iq,kl
    '''
    mol = nevpt_grad.mol
    auxmol = _auxmol(nevpt_grad.nevpt)
    nao, nbas = mol.nao, mol.nbas
    naux = auxmol.nao
    nao_pair = nao * (nao+1) // 2
    aux_loc = auxmol.ao_loc
    wpair = _pair_weights(nao)
    if max_memory is None:
        max_memory = nevpt_grad.max_memory

    int3c, dferi, dfdm2 = solve_df_rdm2(nevpt_grad, fdm2, max_memory)

    Imat = np.zeros((nao, nao))
    de = np.zeros((len(atmlst), 3))
    dvec = np.zeros((3, nao))
    get_int3c_ip1 = _int3c_wrapper(mol, auxmol, 'int3c2e_ip1', 's1')
    blksize = max(1, min(int(max_memory*.3e6/8/max(nao**2*6, 1)), naux))
    for shl0, shl1, nL in balance_partition(aux_loc, blksize):
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        d_blk = lib.unpack_tril(dfdm2[p0:p1])
        c_blk = lib.unpack_tril(int3c[p0:p1])
        Imat += lib.dot(c_blk.reshape(-1,nao).T, d_blk.reshape(-1,nao)) * 2
        c_blk = None
        int3c_ip1 = get_int3c_ip1((0, nbas, 0, nbas, shl0, shl1))
        dvec -= np.einsum('xijp,ijp->xi', int3c_ip1,
                             np.ascontiguousarray(d_blk.transpose(1,2,0))) * 2
        int3c_ip1 = d_blk = None

    aoslices = mol.aoslice_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]
        de[k] += dvec[:,p0:p1].sum(axis=1)

    if nevpt_grad.auxbasis_response:
        aux_de = np.zeros((3, naux))
        dfdm2_w = dfdm2 * wpair
        get_int3c_ip2 = _int3c_wrapper(mol, auxmol, 'int3c2e_ip2', 's2ij')
        blksize = max(1, min(int(max_memory*.3e6/8/max(nao_pair*4, 1)), naux))
        for shl0, shl1, nL in balance_partition(aux_loc, blksize):
            p0, p1 = aux_loc[shl0], aux_loc[shl1]
            int3c_ip2 = get_int3c_ip2((0, nbas, 0, nbas, shl0, shl1))
            aux_de[:,p0:p1] -= np.einsum('xyp,py->xp', int3c_ip2,
                                            dfdm2_w[p0:p1]) * 2
            int3c_ip2 = None
        metric = lib.dot(dferi*wpair, dfdm2.T) * 2
        aux_de += lib.einsum('xpq,pq->xp', auxmol.intor('int2c2e_ip1'), metric)
        auxslices = auxmol.aoslice_by_atom()
        for k, ia in enumerate(atmlst):
            p0, p1 = auxslices[ia,2:]
            de[k] += aux_de[:,p0:p1].sum(axis=1)
    return de, Imat
