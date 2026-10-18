import ctypes
from pyscf import mcscf, ao2mo, fci, lib
from pyscf.cc import ccsd, _ccsd
from pyscf.ao2mo import _ao2mo
from pyscf.grad import mcpdft
from pyscf.lib import logger
from pyscf.fci import cistring
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.mcscf.addons import StateAverageMixFCISolver
import numpy as np
from numpy import einsum
from functools import partial
from functools import reduce
einsum = partial(einsum, optimize=True)

CLASSES = ("0", "+1", "-1", "+2", "-2", "0'", "+1'", "-1'")

from .derivatives import (compute_t1_0, compute_t1_p1, compute_t1_m1, compute_t1_p2, compute_t1_m2, compute_t1_0p, compute_t1_p1p, compute_t1_m1p)
from .semicanonical import get_semicanonical_response
from ..rdm import make_tran_rdm1234

def _symmetrize_rdm2(rdm2):
    neo = rdm2.shape[0]
    rdm2 = np.ascontiguousarray(rdm2)
    pair_matrix = rdm2.reshape(neo**2, neo**2)

    lib.hermi_sum(rdm2.reshape(neo**2, neo, neo), axes=(0, 2, 1), hermi=1, inplace=True)
    lib.inplace_transpose_scale(pair_matrix, alpha=0.125)
    lib.hermi_sum(rdm2.reshape(neo**2, neo, neo), axes=(0, 2, 1), hermi=1, inplace=True)
    lib.hermi_sum(pair_matrix, hermi=1, inplace=True)
    return rdm2

def _make_rdm2_X2(Gamma_SA, Gamma_theta, Gamma_bar, kappa, eri, nocc):
    nmo = Gamma_SA.shape[0]

    so = slice(0, nocc)
    Gamma_SA_occ = np.ascontiguousarray(Gamma_SA[so, so, so, so])
    Gamma = np.ascontiguousarray(Gamma_theta[so, so, so, so] + Gamma_bar[so, so, so, so])
    kappa_occ = kappa[so]

    dtype = np.result_type(Gamma_SA, Gamma_theta, Gamma_bar, kappa, eri)
    Gamma_mo = np.zeros(Gamma_SA.shape, dtype=dtype)
    Gamma_mo[so, so, so, so] = Gamma
    X2 = np.zeros((nmo, nmo), dtype=dtype)
    X2[so] += einsum('ptrs,qtrs->pq', Gamma,
                     eri[:, so, so, so])

    temp = einsum('oqrs,op->pqrs', Gamma_SA_occ, kappa_occ)
    Gamma_mo[:, so, so, so] += temp
    X2 += einsum('ptrs,qtrs->pq', temp, eri[:, so, so, so])
    del temp

    temp = einsum('pors,oq->pqrs', Gamma_SA_occ, kappa_occ)
    Gamma_mo[so, :, so, so] += temp
    X2[so] += einsum('ptrs,qtrs->pq', temp, eri[:, :, so, so])
    del temp

    temp = einsum('pqos,or->pqrs', Gamma_SA_occ, kappa_occ)
    Gamma_mo[so, so, :, so] += temp
    X2[so] += einsum('ptrs,qtrs->pq', temp, eri[:, so, :, so])
    del temp

    temp = einsum('pqro,os->pqrs', Gamma_SA_occ, kappa_occ)
    Gamma_mo[so, so, so, :] += temp
    X2[so] += einsum('ptrs,qtrs->pq', temp, eri[:, so, so, :])
    return Gamma_mo, X2

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

def grad(nevpt_grad, atmlst):
    nevpt = nevpt_grad.nevpt
    for CLASS in nevpt_grad.classes:
        _class_kernel(nevpt_grad, CLASS)
    ncore = nevpt.ncore
    nocc = nevpt.nocc
    nmo = nevpt.nmo
    c = nevpt_grad.c
    a = nevpt_grad.a
    e = nevpt_grad.e
    for name, key in {'aa':(a,a), 'ca':(c,a), 'ce':(c,e), 'ae':(a,e)}.items():
        value = nevpt_grad.h1eff.__dict__[name]
        nevpt_grad.rdm1[key] += value
        # nevpt_grad.rdm2[c, c, key[0], key[1]] += 2 * einsum('pq,ij->ijpq', value, np.eye(ncore))
        # nevpt_grad.rdm2[c, key[0], key[1], c] -= einsum('pq,ij->ipqj', value, np.eye(ncore))
    nevpt_grad.rdm1 = 0.5 * (nevpt_grad.rdm1 + nevpt_grad.rdm1.T)
    # blocks = {
    #     'aaaa': (a,a,a,a), 'aaae': (a,a,a,e), 'aeae': (a,e,a,e), 'caaa': (c,a,a,a), 'caae': (c,a,a,e),
    #     'caca': (c,a,c,a), 'cace': (c,a,c,e), 'ceaa': (c,e,a,a), 'ceae': (c,e,a,e), "cece": (c,e,c,e),
    # }
    # for name, key in blocks.items():
    #     nevpt_grad.rdm2[key] += nevpt_grad.v2e.__dict__[name][:]
    d2 = tuple((nevpt_grad.v2e.__dict__[dkey] for dkey in nevpt_grad.v2e._keys))
    _rdm2_mo2ao(nevpt_grad.nevpt, d2, nevpt_grad.nevpt.mo, nevpt_grad.v2e.fdm2)

    nevpt_grad.mo_energy[c] = nevpt_grad.mo_energy.c
    nevpt_grad.mo_energy[e] = nevpt_grad.mo_energy.e

    converged, Lvec = nevpt_grad.solve_lagrange(mo=nevpt.mo, ci=nevpt.ci, eris=nevpt.eris)[:2]
    if not converged:
        nevpt.log.warn('Z-vector equation did not converge!')

    mf = nevpt.mf
    mc = nevpt_grad.mc
    mc_grad = mc.Gradients()
    Lorb, Lci = mc_grad.unpack_uniq_var(Lvec)
    kappa = -Lorb

    gamma_SA, Gamma_SA = mc.fcisolver.make_rdm12(nevpt.ci, mc.ncas, mc.nelecas)
    gamma_SA, Gamma_SA = mcscf.addons._make_rdm12_on_mo(gamma_SA, Gamma_SA, mc.ncore, mc.ncas, nmo)
    gamma_theta, Gamma_theta = mc.fcisolver.states_make_rdm12(nevpt.ci, mc.ncas, mc.nelecas)
    gamma_theta, Gamma_theta = gamma_theta[nevpt_grad.state], Gamma_theta[nevpt_grad.state]
    gamma_theta, Gamma_theta = mcscf.addons._make_rdm12_on_mo(gamma_theta, Gamma_theta, mc.ncore, mc.ncas, nmo)
    gamma_tilde = (
        + einsum('oq,op->pq', gamma_SA, kappa)
        - einsum('po,qo->pq', gamma_SA, kappa)
    )
    # Gamma_tilde = (
    #     + einsum('oqrs,op->pqrs', Gamma_SA, kappa)
    #     + einsum('pors,oq->pqrs', Gamma_SA, kappa)
    #     + einsum('pqos,or->pqrs', Gamma_SA, kappa)
    #     + einsum('pqro,os->pqrs', Gamma_SA, kappa)
    # )
    gamma_bar, Gamma_bar = mc.fcisolver.trans_rdm12(Lci, nevpt.ci, mc.ncas, mc.nelecas)
    gamma_bar += gamma_bar.T
    Gamma_bar += Gamma_bar.transpose(3,2,1,0)
    def _make_rdm12_on_mo(casdm1, casdm2, ncore, ncas, nmo):
        nocc = ncas + ncore
        dm1 = np.zeros((nmo,nmo))
        dm1[ncore:nocc,ncore:nocc] = casdm1
    
        dm2 = np.zeros((nmo,nmo,nmo,nmo))
        dm2[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc] = casdm2
        for i in range(ncore):
            dm2[i,i,ncore:nocc,ncore:nocc] = dm2[ncore:nocc,ncore:nocc,i,i] =2*casdm1
            dm2[i,ncore:nocc,ncore:nocc,i] = dm2[ncore:nocc,i,i,ncore:nocc] = -casdm1
        return dm1, dm2
    gamma_bar, Gamma_bar = _make_rdm12_on_mo(gamma_bar, Gamma_bar, mc.ncore, mc.ncas, nmo)
    gamma_mo = gamma_theta + gamma_tilde + gamma_bar
    # Gamma_mo = Gamma_theta + Gamma_tilde + Gamma_bar
    mo = nevpt.mo
    hcore = einsum('ij,ip,jq->pq', mf.get_hcore(), mo, mo)
    if getattr(mf, 'with_df', None) is None:
        eri = ao2mo.kernel(mf._eri, mo, compact=False).reshape([nmo]*4)
    else:
        eri = mf.with_df.ao2mo(mo, compact=False).reshape([nmo]*4)
    Gamma_mo, X2_mo = _make_rdm2_X2(Gamma_SA, Gamma_theta, Gamma_bar, kappa, eri, nocc)
    # X_mo = einsum('pr,qr->pq', gamma_mo, hcore) + einsum('ptrs,qtrs->pq', Gamma_mo, eri)
    X_mo = einsum('pr,qr->pq', gamma_mo, hcore) + X2_mo
    eri = None
    gamma_mo += nevpt_grad.rdm1
    blocks = {
        'aaaa': (a,a,a,a), 'aaae': (a,a,a,e), 'aeae': (a,e,a,e), 'caaa': (c,a,a,a), 'caae': (c,a,a,e),
        'caca': (c,a,c,a), 'cace': (c,a,c,e), 'ceaa': (c,e,a,a), 'ceae': (c,e,a,e), "cece": (c,e,c,e),
    }
    for name, key in blocks.items():
        Gamma_mo[key] += 2 * nevpt_grad.v2e.__dict__[name][:]

    D = np.zeros((nmo,nmo))
    D[c,c] = 2 * np.eye(ncore)
    if nevpt_grad.canonstep == 0 or nevpt_grad.canonstep == 1:
        D[a,a] = nevpt.casdm1
    elif nevpt_grad.canonstep == 2:
        D[a,a] = nevpt.sa_casdm1
    else:
        raise NotImplementedError
    for p, mo_energy in enumerate(nevpt_grad.mo_energy):
        Gamma_mo[p,p,:,:] += 2 * mo_energy * D
        Gamma_mo[p,:,:,p] -= mo_energy * D
    for name, key in {'aa':(a,a), 'ca':(c,a), 'ce':(c,e), 'ae':(a,e)}.items():
        value = nevpt_grad.h1eff.__dict__[name]
        Gamma_mo[c, c, key[0], key[1]] += 4 * einsum('pq,ij->ijpq', value, np.eye(ncore))
        Gamma_mo[c, key[0], key[1], c] -= 2 * einsum('pq,ij->ipqj', value, np.eye(ncore))
    Gamma_mo += 2 * einsum('pq,rs->pqrs', nevpt_grad._X_semi, nevpt_grad._D_semi)
    Gamma_mo -= einsum('pq,rs->prsq', nevpt_grad._X_semi, nevpt_grad._D_semi)
    X_mo += nevpt_grad.rdm1e * 0.5
    rdm1e_ao = einsum('pq,ip,jq->ij', X_mo, mo, mo)
    rdm1_ao = einsum('pq,ip,jq->ij', gamma_mo, mo, mo)
    rdm1_ao = 0.5 * (rdm1_ao + rdm1_ao.T)
    rdm1e_ao = 0.5 * (rdm1e_ao + rdm1e_ao.T)
    rdm2_ao = einsum('pqrs,ip,jq,kr,ls->ijkl', Gamma_mo, mo, mo, mo, mo)
    # rdm2_ao += rdm2_ao.transpose(0,1,3,2)
    # rdm2_ao += rdm2_ao.transpose(1,0,2,3)
    # rdm2_ao += rdm2_ao.transpose(2,3,0,1)
    # rdm2_ao *= 0.125
    rdm2_ao = _symmetrize_rdm2(rdm2_ao)
    return rdm1_ao, rdm2_ao, rdm1e_ao

class Gradients(mcpdft.Gradients):
    _keys = ['verbose', 'mol', 'nevpt', 'base', 'max_memory', 'mc', 'canonstep', 'unit', 'atmlst', 'de', 'classes', 'excluded_classes', 'state', 'e_mcscf', 'weights', 'nroots', 'ngorb', 'spin_states', 'na_states', 'nb_states', 'nci', 'nlag', 'level_shift', 'conv_atol', 'conv_rtol', 'max_cycle', '_t1', 'c', 'a', 'e', 'mo_energy', 'h1eff', 'v2e', 'rdms', 'rdm1', 'rdm2', 'rdm1e']

    def __init__(self, nevpt):
        self.verbose = nevpt.verbose
        self.mol = nevpt.mol
        self.nevpt = nevpt
        self.base = self.nevpt.base
        self.max_memory = self.nevpt.max_memory
        self.mc = self.nevpt.mc
        self.canonstep = self.nevpt.canonstep
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
        # self.rdm2 = np.zeros([self.nevpt.nmo]*4)
        self.rdm1e = np.zeros([self.nevpt.nmo]*2)
        self.eris = self.nevpt.eris

    def get_wfn_response(self, **kwargs):
        nevpt = self.nevpt
        nmo = nevpt.nmo
        ncore = nevpt.ncore
        c = self.c
        a = self.a
        e = self.e

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
        Imat = np.zeros((nmo,nmo))
        if getattr(nevpt.mc, 'with_df', None) is None:
            eri_ao = ao2mo.restore(4, nevpt.mf._eri, nmo)
            diagidx = np.arange(nmo)
            diagidx = diagidx*(diagidx+1)//2 + diagidx
            max_memory = max(0, nevpt.max_memory - lib.current_memory()[0])
            blksize = max(1, int(max_memory*.9e6/8/(nmo**3*2.5)))
            blksize = min(nmo, blksize)
            for ip0, ip1 in lib.prange(0, nmo, blksize):
                dm2buf = _load_block_tril(
                    self.v2e.fdm2['dm2'], ip0, ip1, nmo)
                dm2buf[:,:,diagidx] *= .5
                eri0 = _load_block_tril(eri_ao, ip0, ip1, nmo)
                Imat += 2 * einsum('ipx,iqx->pq', eri0, dm2buf)
                eri0 = dm2buf = None
            eri_ao = None
        else:
            eri = nevpt.mc.with_df.ao2mo(
                nevpt.mo, compact=False).reshape([nmo]*4)
            blocks = {
                'aaaa': (a,a,a,a), 'aaae': (a,a,a,e), 'aeae': (a,e,a,e), 'caaa': (c,a,a,a), 'caae': (c,a,a,e),
                'caca': (c,a,c,a), 'cace': (c,a,c,e), 'ceaa': (c,e,a,a), 'ceae': (c,e,a,e), "cece": (c,e,c,e),
            }
            for name, key in blocks.items():
                p, q, r, s = key
                rdm2 = self.v2e.__dict__[name][:]
                gorb[:,p] += einsum('tqrs,pqrs->tp', eri[:,q,r,s], rdm2)
                gorb[:,q] += einsum('ptrs,pqrs->tq', eri[p,:,r,s], rdm2)
                gorb[:,r] += einsum('pqts,pqrs->tr', eri[p,q,:,s], rdm2)
                gorb[:,s] += einsum('pqrt,pqrs->ts', eri[p,q,r,:], rdm2)
            eri = rdm2 = None
        Imat += 2 * lib.dot(veff[0], dm_list[1])
        Imat += 2 * lib.dot(veff[1], dm_list[0])
        Imat += 2 * lib.dot(veff[2], dm_list[3])
        Imat += 2 * lib.dot(veff[3], dm_list[2])
        gorb += reduce(lib.dot, (nevpt.mo.T, Imat, nevpt.mf.get_ovlp(), nevpt.mo))
        gorb_semi, gci_semi, rdm1_semi, X_semi, D_semi = get_semicanonical_response(self, gorb)
        self.rdm1 += rdm1_semi
        self._X_semi = X_semi
        self._D_semi = D_semi
    
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
        return bvec

    def project_Aop(self, Aop, ci, state):
        Aop = super().project_Aop(Aop, ci, state)
        def my_Aop(x):
            projected_x = self._project_hessian_vector(x)
            Ax = Aop(projected_x)
            Ax = self._project_hessian_vector(Ax)
            Ax += (x - projected_x)
            return Ax
        return my_Aop

    def get_lagrange_precond(self, Adiag, level_shift=None, **kwargs):
        precond = super().get_lagrange_precond(Adiag, level_shift=level_shift, **kwargs)
        def my_precond(x):
            projected_x = self._project_hessian_vector(x)
            Ax = precond(projected_x)
            Ax = self._project_hessian_vector(Ax)
            Ax += (x - projected_x)
            return Ax
        return my_precond

    def get_init_guess(self, bvec, Adiag, Aop, precond):
        '''Initial guess should solve the problem for SA-SA rotations'''
        sing_tol = getattr(self, 'sing_tol_sasa', 1e-8)
        ci = self.nevpt.ci
        state = self.state
        if self.nroots == 1 and isinstance(ci, np.ndarray): ci = [ci,]
        idx_spin = [i for i in range (self.nroots)
                    if self.spin_states[i]==self.spin_states[state]]
        ci_blk = np.asarray([ci[i].ravel() for i in idx_spin])
        b_orb, b_ci = self.unpack_uniq_var(bvec)
        b_ci_blk = np.asarray([b_ci[i].ravel() for i in idx_spin])
        x0 = np.zeros_like(bvec)
        if self.nroots > 1:
            b_sa = np.dot(ci_blk.conjugate(), b_ci[state].ravel())
            A_sa = 2 * self.weights[state] * (self.e_mcscf
                - self.e_mcscf[state])
            idx_valid = np.abs(A_sa) >= sing_tol
            x0_sa = np.zeros_like(b_sa)
            x0_sa[idx_valid] = -b_sa[idx_valid] / A_sa[idx_valid]

            ovlp = ci_blk.conjugate() @ b_ci_blk.T
            logger.debug(self, f'Linear response SA-SA part:\n{ovlp}')
            logger.debug(self, f'Linear response SA-CI norms:\n{np.linalg.norm(b_ci_blk.T - ci_blk.T @ ovlp, axis=1)}')
            if self.ngorb: logger.debug (self, 'Linear response orbital '
                f'norms:\n{np.linalg.norm (bvec[:self.ngorb])}')
            logger.debug(self, 'SA-SA Lagrange multiplier for root '
                f'{state}:\n{x0_sa}')
            x0_orb, x0_ci = self.unpack_uniq_var(x0)
            x0_ci[state] = np.dot(x0_sa, ci_blk).reshape(
                self.na_states[state], self.nb_states[state])
            x0 = self.pack_uniq_var (x0_orb, x0_ci)
        r0 = bvec + Aop(x0)
        r0_orb, r0_ci = self.unpack_uniq_var(r0)
        r0_ci_blk = np.asarray([r0_ci[i].ravel() for i in idx_spin])
        ovlp = ci_blk.conjugate() @ r0_ci_blk.T
        logger.debug(self, 'Lagrange residual SA-SA part after solving SA-SA'
            f' part:\n{ovlp}')
        logger.debug(self, 'Lagrange residual SA-CI norms after solving SA-SA'
            ' part:\n{}'.format(np.linalg.norm (r0_ci_blk.T - ci_blk.T @ ovlp,
            axis=1)))
        if self.ngorb: logger.debug (self, 'Lagrange residual orbital norms '
            'after solving SA-SA part:\n{}'.format(np.linalg.norm (
                r0_orb)))
        x0 += precond(-r0)
        r1_orb, r1_ci = self.unpack_uniq_var(r0)
        r1_ci_blk = np.asarray([r1_ci[i].ravel () for i in idx_spin])
        ovlp = ci_blk.conjugate() @ r1_ci_blk.T
        logger.debug(self, 'Lagrange residual SA-SA part after first '
            f'precondition:\n{ovlp}')
        logger.debug(self, 'Lagrange residual SA-CI norms after first '
            'precondition:\n{}'.format(np.linalg.norm (r1_ci_blk.T - ci_blk.T @ ovlp,
            axis=1)))
        if self.ngorb: logger.debug (self, 'Lagrange residual orbital norms '
            'after first precondition:\n{}'.format(np.linalg.norm (
                r1_orb)))
        return self._project_hessian_vector(x0)

    def kernel(self, atmlst=None):
        cput0 = (logger.process_clock(), logger.perf_counter())

        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst
        if atmlst is None:
            atmlst = list(range(self.mol.natm))

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

    def _finalize(self):
        return

    def _hessian_gauge(self, ci=None, state=None):
        if ci is None: ci = self.nevpt.ci
        if state is None: state = self.state
        if self.nroots == 1 and isinstance(ci, np.ndarray): ci = [ci]

        casdm1 = self.nevpt.sa_casdm1.copy()
        natocc, natorb = np.linalg.eigh(casdm1)
        zero_occ = np.where(abs(natocc) < 1e-6)[0]
        full_occ = np.where(abs(2 - natocc) < 1e-6)[0]

        weights = np.atleast_1d(self.weights)
        e_mcscf = np.atleast_1d(self.e_mcscf)
        sing_tol = getattr(self, 'sing_tol_sasa', 1e-8)
        A_sa = 2 * weights[state] * (e_mcscf - e_mcscf[state])
        project_ci = np.zeros((self.nroots, self.nroots), dtype=bool)
        for root in range(self.nroots):
            for ref_root in range(self.nroots):
                if self.spin_states[root] != self.spin_states[ref_root]:
                    continue
                retain_target_rotation = (
                    root == state
                    and ref_root != state
                    and abs(A_sa[ref_root]) > sing_tol
                )
                project_ci[root, ref_root] = not retain_target_rotation

        self._hessian_gauge_ci = ci
        self._hessian_gauge_state = state
        self._hessian_zero_active = natorb[:, zero_occ]
        self._hessian_zero_occ = natocc[zero_occ]
        self._hessian_full_active = natorb[:, full_occ]
        self._hessian_full_occ = natocc[full_occ]
        self._hessian_project_ci = project_ci
        self._hessian_gauge_dim = (
            + self._hessian_zero_active.shape[1] * self.nevpt.nextern
            + self._hessian_full_active.shape[1] * self.nevpt.ncore
            + np.count_nonzero(project_ci)
        )
        self.nevpt.log.debug('Hessian gauge dimension = %d, full active occupations = %s, zero active occupations = %s', self._hessian_gauge_dim, self._hessian_full_occ, self._hessian_zero_occ)

        return self

    def _project_hessian_vector(self, vec):
        xorb, xci = self.unpack_uniq_var(vec)
        xorb = np.array(xorb)
        xci = [np.array(xci_) for xci_ in xci]

        c = self.c
        a = self.a
        e = self.e
        zero_active = self._hessian_zero_active
        full_active = self._hessian_full_active
        if zero_active.shape[1] and self.nevpt.nextern:
            redundant = reduce(lib.dot, (zero_active, zero_active.T, xorb[a,e]))
            xorb[a,e] -= redundant
            xorb[e,a] += redundant.T
        if full_active.shape[1] and self.nevpt.ncore:
            redundant = reduce(lib.dot, (full_active, full_active.T, xorb[a,c]))
            xorb[a,c] -= redundant
            xorb[c,a] += redundant.T

        ref_ci = self._hessian_gauge_ci
        for root in range(self.nroots):
            for ref_root in range(self.nroots):
                if not self._hessian_project_ci[root, ref_root]:
                    continue
                xci[root] -= einsum('ij,ij', ref_ci[ref_root], xci[root]) * ref_ci[ref_root]

        return self.pack_uniq_var(xorb, xci)

    def _project_bvec(self, bvec):
        ci = self.nevpt.ci
        state = self.state
        if self.nroots == 1 and isinstance(ci, np.ndarray): ci = [ci]

        sing_tol = getattr(self, 'sing_tol_sasa', 1e-8)
        borb, bci = self.unpack_uniq_var(bvec)
        e_mcscf = np.atleast_1d(self.e_mcscf)
        for root in range(self.nroots):
            if root == state or self.spin_states[root] != self.spin_states[self.state]:
                continue
            if abs(e_mcscf[root] - e_mcscf[state]) > sing_tol:
                continue
            comp = einsum('ij,ij', ci[root], bci[self.state])
            if abs(comp) > 1e-8:
                self.nevpt.log.warn('''
    The target SA-CASSCF root %d is degenerate with root %d,
    but the FIC-NEVPT2 Lagrange vector along this state rotation
    if %.12g. A state-specific gradient is not uniquely defined!''', state, root, abs(comp))

        projected_bvec = self._project_hessian_vector(bvec)
        if np.linalg.norm(bvec - projected_bvec) >= 1e-8:
            self.nevpt.log.warn('''
    Lagrange vector is not orthogonal to the SA-CASSCF Hessian gauge space!
    The residue = %.12f. A state-specific gradient is not uniquely defined!''', np.linalg.norm(bvec - projected_bvec))

        return projected_bvec

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

def _load_block_tril(h5dat, row0, row1, nao, out=None):
    nao_pair = nao * (nao+1) // 2
    if out is None:
        out = np.ndarray((row1-row0,nao,nao_pair))
    dat = h5dat[row0*(row0+1)//2:row1*(row1+1)//2]
    p1 = 0
    for i in range(row0, row1):
        p0, p1 = p1, p1 + i+1
        out[i-row0,:i+1] = dat[p0:p1]
        for j in range(row0, i):
            out[j-row0,i] = out[i-row0,j]
    for i in range(row1, nao):
        i2 = i*(i+1)//2
        out[:,i] = h5dat[i2+row0:i2+row1]
    return out