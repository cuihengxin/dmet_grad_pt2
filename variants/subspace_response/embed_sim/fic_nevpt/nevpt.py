from . import integrals, rdm
from .amplitudes import (compute_t1_0, compute_t1_p1, compute_t1_m1, compute_t1_p2, compute_t1_m2, compute_t1_0p, compute_t1_p1p, compute_t1_m1p)

from pyscf import lib
import numpy as np
from numpy import einsum
from functools import partial
try:
    from pytblis import einsum
except ImportError:
    einsum = partial(einsum, optimize=True)

FRAC_OCC_THRESHOLD = 1e-6
CLASSES = ("0", "+1", "-1", "+2", "-2", "0'", "+1'", "-1'")

def _class_kernel(nevpt, CLASS):
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
    return _class_kernels[CLASS](nevpt)

def kernel(nevpt):
    nevpt.make_rdm1234()
    nevpt.get_integrals()
    for CLASS in nevpt.classes:
        result = _class_kernel(nevpt, CLASS)
        nevpt._e_corr[CLASS], nevpt._t1[CLASS], nevpt._koopmans[CLASS], nevpt._overlap[CLASS], nevpt._overlap_discard[CLASS], nevpt._V1[CLASS], nevpt._ic_coeff[CLASS], nevpt._t1_[CLASS]  = result
    nevpt.e_corr = sum(nevpt._e_corr.values())
    nevpt.e_tot = nevpt.e_ref + nevpt.e_corr
    nevpt.log.note('E(NEVPT2) = %.16g E_corr_pt = %.16g', nevpt.e_tot, nevpt.e_corr)
    return nevpt.e_tot, nevpt.e_corr

class NEVPT:
    _keys = ['mc', 'base', 'state', 'mf', '_scf', 'mol', 'mo', 'nmo', 'ncore', 'ncas', 'nocc', 'nextern', 'nelecas', 'ci', 'state_ci', 'nstates', 'e_ref', 'eris', 'sa_casdm1', 'casdm1', 'canonstep', 'sa_fock_ao', 'sa_fock', 'fock_ao', 'fock', 'mo_energy', 'mo_coeff', 'classes', 'excluded_classes', 'overlap_threshold', 'verbose', 'stdout', 'log', 'max_memory', 'rdms', 'h1e', 'v2e', 'h1eff', 'e_corr', 'e_tot']

    def __init__(self, mc, state=0, canonstep=1, overlap_threshold=1e-8, classes=None, verbose=None):
        self.mc = self.base = mc
        self.state = state
        self.mf = self._scf = self.mc._scf
        self.mol = self.mc.mol
        self.nmo = self.mc.mo_coeff.shape[0]
        self.ncore = self.mc.ncore
        self.ncas = self.mc.ncas
        self.nocc = self.ncore + self.ncas
        self.nextern = self.nmo - self.nocc
        self.nelecas = self.mc.nelecas
        self.ci = self.mc.ci
        self.state_ci = self.ci[self.state]
        self.nstates = len(self.ci)
        self.e_ref = self.mc.e_states[self.state]
        self.sa_casdm1 = self.mc.fcisolver.make_rdm1(self.ci, self.ncas, self.nelecas)
        self.casdm1 = self.mc.fcisolver.undo_state_average().make_rdm1(self.state_ci, self.ncas, self.nelecas)
        self.verbose = self.mc.verbose if verbose is None else verbose
        self.stdout = self.mol.stdout
        self.log = lib.logger.new_logger(self.mol, self.verbose)
        self.max_memory = self.mol.max_memory

        self.canonstep = canonstep
        self._canonicalize(self.canonstep, self.ci, self.state_ci, self.sa_casdm1, self.casdm1)
        self.mo_energy = lib.tag_array(
            self.mo_energy,
            c = self.mo_energy[:self.ncore],
            a = self.mo_energy[self.ncore:self.nocc],
            e = self.mo_energy[self.nocc:]
        )
        self.mo = lib.tag_array(
            self.mo,
            c = self.mo[:, :self.ncore],
            a = self.mo[:, self.ncore:self.nocc],
            e = self.mo[:, self.nocc:]
        )
        self.mo_coeff = self.mo
        self.classes = CLASSES if classes is None else classes
        self.excluded_classes = tuple(excluded_class for excluded_class in CLASSES if excluded_class not in self.classes)
        self.overlap_threshold = overlap_threshold

        self.h1e = None
        self.h1eff = None
        self.v2e = None
        self.rdms = None
        self._koopmans = {}
        self._overlap = {}
        self._overlap_discard = {}
        self._V1 = {}
        self._t1 = {}
        self._ic_coeff = {}
        self._t1_ = {}
        self._e_corr = {}
        self.e_corr = None
        self.e_tot = None

    make_rdm1234 = rdm.make_rdm1234
    get_integrals = integrals.transform_integrals

    def dump_flags(self):
        self.log.info('')
        self.log.info('******** %s ********', self.__class__)
        self.log.info('ncore = %s, ncas = %s, nextern = %s', self.ncore, self.ncas, self.nextern)
        self.log.info('state = %s', self.state)
        self.log.info('canonstep = %s', self.canonstep)
        self._mem_usage()
        self.log.info('')
        return self

    def _mem_usage(self):
        n = {'c': self.ncore, 'a': self.ncas, 'e': self.nextern}
        keys = ['aaaa', 'aaae', 'aeae', 'caaa', 'caae', 'caca', 'cace', 'ceaa', 'ceae', 'cece']
        incore = 0
        for key in keys:
            key1, key2, key3, key4 = key
            n1, n2, n3, n4 = n[key1], n[key2], n[key3], n[key4]
            incore += n1 * n2 * n3 * n4
        incore = incore * 8 / 1e9
        mem_now = lib.current_memory()[0] / 1e3
        max_mem = self.max_memory*.9 / 1e3 - mem_now
        if incore > max_mem:
            self.log.warn('memory needed = %.5g GB > max_memory = %.5g GB', incore, max_mem)
        else:
            self.log.debug('memory needed = %.5g GB < max_memory = %.5g GB', incore, max_mem)
        return self

    def _canonicalize(self, canonstep=None, ci=None, state_ci=None, sa_casdm1=None, casdm1=None):
        if canonstep is None: canonstep = self.canonstep
        if ci is None: ci = self.ci
        if state_ci is None: state_ci = self.state_ci
        if sa_casdm1 is None: sa_casdm1 = self.sa_casdm1
        if casdm1 is None: casdm1 = self.casdm1

        mc = self.mc
        ncore = self.ncore
        nocc = self.nocc
        nmo = self.nmo
        mo, ci, mo_occ = mc.cas_natorb(mc.mo_coeff, mc.ci, eris=None, sort=False, casdm1=sa_casdm1, verbose=0)
        cas_occ = np.asarray(mo_occ[ncore:nocc])
        full_act = ncore + np.where(2 - cas_occ < FRAC_OCC_THRESHOLD)[0]
        zero_act = ncore + np.where(cas_occ < FRAC_OCC_THRESHOLD)[0]

        self.ci = ci
        self.state_ci = ci[self.state]
        self.sa_casdm1 = mc.fcisolver.make_rdm1(ci, self.ncas, self.nelecas)
        self.casdm1 = mc.fcisolver.undo_state_average().make_rdm1(self.state_ci, self.ncas, self.nelecas)
        self.sa_fock_ao = self.mc.get_fock(mo, self.ci, None, self.sa_casdm1)
        self.fock_ao = self.mc.get_fock(mo, self.state_ci, None, self.casdm1)
        if self.canonstep == 1:
            occ2_idx = np.append(np.arange(ncore), full_act)
            vir2_idx = np.append(zero_act, np.arange(nocc, nmo))
            mo = _diag_subfock(mc, mo, self.fock_ao, occ2_idx)
            mo = _diag_subfock(mc, mo, self.fock_ao, vir2_idx)
        self.mo = mo
        self.eris = mc.ao2mo(self.mo)
        self.sa_fock = einsum('ip,ij,jq->pq', self.mo, self.sa_fock_ao, self.mo)
        self.fock = einsum('ip,ij,jq->pq', self.mo, self.fock_ao, self.mo)
        if self.canonstep == 0 or self.canonstep == 1:
            self.mo_energy = np.diagonal(self.fock)
        elif self.canonstep == 2:
            self.mo_energy = np.diagonal(self.sa_fock)
        else:
            raise NotImplementedError
        self._full_act = full_act
        self._zero_act = zero_act
        return self

    def ao2mo(self, mo_coeff=None):
        return self.mc.ao2mo(mo_coeff)

    def pack_uniq_var(self, mat):
        return self.mc.pack_uniq_var(mat)

    def unpack_uniq_var(self, v):
        return self.mc.unpack_uniq_var(v)

    def kernel(self):
        self.dump_flags()
        return kernel(self)

    def nuc_grad_method(self):
        from . import grad
        return grad.Gradients(self)

    Gradients = nuc_grad_method

def _diag_subfock(mc, mo, fock_ao, idx):
    if idx.size <= 1:
        return mo
    if getattr(mo, 'orbsym', None) is not None:
        orbsym = mo.orbsym
    else:
        orbsym = np.zeros(mo.shape[1], dtype=int)
    extrasym = getattr(mc, 'extrasym', None)
    if extrasym is not None:
        orbsym = np.asarray([
            str(i1) + str(i2) for i1, i2 in zip(orbsym, extrasym)
        ])
    c = mo[:,idx]
    fock = einsum('ip,ij,jq->pq', c, fock_ao, c)
    _, u = mc._eig(fock, None, None, orbsym[idx])
    mo[:,idx] = c @ u
    return mo