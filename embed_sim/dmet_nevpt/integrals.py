from types import SimpleNamespace
from pyscf import ao2mo, lib
from numpy import einsum
from functools import partial
einsum = partial(einsum, optimize=True)

class _ERIS:
    def __init__(self):
        self.aaaa = None
        self.aaae = None
        self.aeae = None
        self.caaa = None
        self.caae = None
        self.caca = None
        self.cace = None
        self.ceaa = None
        self.ceae = None
        self.cece = None

def transform_2e_chem(nevpt, mo, key):
    n = {'c': nevpt.ncore, 'a': nevpt.ncas, 'e': nevpt.nextern}
    mo = {'c': mo.c, 'a': mo.a, 'e': mo.e}
    key1, key2, key3, key4 = key
    mo1, mo2, mo3, mo4 = mo[key1], mo[key2], mo[key3], mo[key4]
    n1, n2, n3, n4 = n[key1], n[key2], n[key3], n[key4]
    if hasattr(nevpt.mc, 'with_df'):
        eri = nevpt.mc.with_df.ao2mo([mo1, mo2, mo3, mo4], compact=False).reshape([n1, n2, n3, n4])
    else:
        eri = ao2mo.kernel(nevpt.mf._eri, [mo1, mo2, mo3, mo4], compact=False).reshape([n1, n2, n3, n4])
    return eri

def get_h1eff(h1e, v2e_ccpq, v2e_cpqc):
    h1eff = h1e + 2*einsum('ccpq->pq', v2e_ccpq) - einsum('cpqc->pq', v2e_cpqc)
    return h1eff

def transform_integrals(nevpt):
    mc = nevpt.mc
    mo = nevpt.mo
    c = slice(0, nevpt.ncore)
    a = slice(nevpt.ncore, nevpt.nocc)
    e = slice(nevpt.nocc, nevpt.nmo)
    if nevpt.h1e is None:
        h1e = einsum('ip,ij,jq->pq', mo, mc.get_hcore(), mo)
        nevpt.h1e = h1e
    if nevpt.v2e is None:
        v2e = _ERIS()
        v2e.aaaa = transform_2e_chem(nevpt, mo, 'aaaa')
        v2e.aaae = transform_2e_chem(nevpt, mo, 'aaae')
        v2e.aeae = transform_2e_chem(nevpt, mo, 'aeae')
        v2e.caaa = transform_2e_chem(nevpt, mo, 'caaa')
        v2e.caae = transform_2e_chem(nevpt, mo, 'caae')
        v2e.caca = transform_2e_chem(nevpt, mo, 'caca')
        v2e.cace = transform_2e_chem(nevpt, mo, 'cace')
        v2e.ceaa = transform_2e_chem(nevpt, mo, 'ceaa')
        v2e.ceae = transform_2e_chem(nevpt, mo, 'ceae')
        v2e.cece = transform_2e_chem(nevpt, mo, 'cece')
        nevpt.v2e = v2e
    if nevpt.h1eff is None:
        core_dm = lib.dot(mo.c, mo.c.T) * 2
        h1eff = mc.get_hcore() + mc.get_veff(mc.mol, core_dm)
        h1eff = einsum('ip,ij,jq->pq', mo, h1eff, mo)
        h1eff = lib.tag_array(
            h1eff,
            aa = h1eff[a,a],
            ca = h1eff[c,a],
            ce = h1eff[c,e],
            ae = h1eff[a,e]
        )
        nevpt.h1eff = h1eff
    return nevpt