from . import integrals, rdm
from .nevpt import NEVPT
from .amplitudes import (compute_t1_0, compute_t1_p1, compute_t1_m1, compute_t1_p2, compute_t1_m2, compute_t1_0p, compute_t1_p1p, compute_t1_m1p)

from pyscf import lib
import numpy as np
from numpy import einsum
from functools import partial
einsum = partial(einsum, optimize=True)

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

class DFNEVPT(NEVPT):
    _keys = ['mc', 'with_df', 'base', 'state', 'mf', '_scf', 'mol', 'mo', 'nmo', 'ncore', 'ncas', 'nocc', 'nextern', 'nelecas', 'ci', 'state_ci', 'nstates', 'e_ref', 'eris', 'sa_casdm1', 'casdm1', 'canonstep', 'sa_fock_ao', 'sa_fock', 'fock_ao', 'fock', 'mo_energy', 'mo_coeff', 'classes', 'excluded_classes', 'overlap_threshold', 'verbose', 'stdout', 'log', 'max_memory', 'rdms', 'h1e', 'v2e', 'h1eff', 'e_corr', 'e_tot']

    def __init__(self, mc, state=0, canonstep=1, overlap_threshold=1e-8, classes=None, verbose=None):
        self.with_df = getattr(mc, 'with_df', None)
        if self.with_df is None:
            raise ValueError
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
        return grad.DFGradients(self)

    Gradients = nuc_grad_method

NEVPT = DFNEVPT