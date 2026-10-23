from pyscf import fci
import numpy as np

class _RDMS:
    def __init__(self):
        self.ca = None
        self.ccaa = None
        self.cccaaa = None
        self.ccccaaaa = None

def make_tran_rdm1234(bra, ket, ncas, nelecas):
    if np.sum(nelecas) > 1:
        dm1, dm2, dm3, dm4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf', bra, ket, ncas, nelecas)
        dm1, dm2, dm3, dm4 = fci.rdm.reorder_dm1234(dm1, dm2, dm3, dm4)
        dm1 = np.ascontiguousarray(dm1.T)
        dm2 = np.ascontiguousarray(dm2.transpose(0,2,1,3))
        dm3 = np.ascontiguousarray(dm3.transpose(0,2,4,1,3,5))
        dm4 = np.ascontiguousarray(dm4.transpose(0,2,4,6,1,3,5,7))
    else:
        dm1, dm2, dm3 = fci.rdm.make_dm123('FCI3pdm_kern_sf', bra, ket, ncas, nelecas)
        dm1, dm2, dm3 = fci.rdm.reorder_dm123(dm1, dm2, dm3)
        dm1 = np.ascontiguousarray(dm1.T)
        dm2 = np.ascontiguousarray(dm2.transpose(0,2,1,3))
        dm3 = np.ascontiguousarray(dm3.transpose(0,2,4,1,3,5))
        dm4 = np.zeros((ncas,) * 8)
    return dm1, dm2, dm3, dm4

def make_rdm1234(nevpt):
    nevpt.rdms = rdms = _RDMS()
    if nevpt.ncas == 0:
        nevpt.rdms.ca = np.zeros((0,) * 2)
        nevpt.rdms.ccaa = np.zeros((0,) * 4)
        nevpt.rdms.cccaaa = np.zeros((0,) * 6)
        nevpt.rdms.ccccaaaa = np.zeros((0,) * 8)
    else:
        dm1, dm2, dm3, dm4 = make_tran_rdm1234(nevpt.state_ci, nevpt.state_ci, nevpt.ncas, nevpt.nelecas)
        nevpt.rdms.ca = dm1
        nevpt.rdms.ccaa = dm2
        nevpt.rdms.cccaaa = dm3
        nevpt.rdms.ccccaaaa = dm4
    return nevpt