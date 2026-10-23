from pyscf.lib import logger
from pyscf import ao2mo, fci, lib
import numpy as np
from numpy import einsum
from functools import partial
from functools import reduce
einsum = partial(einsum, optimize=True)

NUMERICAL_ZERO = 1e-8

def get_semicanonical_response(nevpt_grad, gorb):
    nevpt = nevpt_grad.nevpt
    ncore = nevpt.ncore
    ncas = nevpt.ncas
    nocc = nevpt.nocc
    nmo = nevpt.nmo
    if nevpt.canonstep == 0 or nevpt.canonstep == 2:
        mo_energy = np.diag(nevpt.sa_fock)
    elif nevpt.canonstep == 1:
        mo_energy = nevpt.mo_energy
    else:
        raise NotImplementedError
    c = nevpt_grad.c
    a = nevpt_grad.a

    Y = gorb - gorb.T
    Z = np.zeros((nmo,nmo))
    occ2_idx = np.append(np.arange(ncore), nevpt._full_act)
    vir2_idx = np.append(nevpt._zero_act, np.arange(nocc, nmo))
    for pi, p in enumerate(occ2_idx):
        for q in occ2_idx[:pi]:
            gap = mo_energy[p] - mo_energy[q]
            if abs(gap) < NUMERICAL_ZERO and abs(Y[p,q]) > NUMERICAL_ZERO:
                nevpt.log.debug('Degeneracy in core-core block')
            elif abs(gap) < NUMERICAL_ZERO and abs(Y[p,q]) <= NUMERICAL_ZERO:
                continue
            else:
                Z[p,q] = -Y[p,q] / gap
    for pi, p in enumerate(vir2_idx):
        for q in vir2_idx[:pi]:
            gap = mo_energy[p] - mo_energy[q]
            if abs(gap) < NUMERICAL_ZERO and abs(Y[p,q]) > NUMERICAL_ZERO:
                nevpt.log.debug('Degeneracy in extern-extern block')
            elif abs(gap) < NUMERICAL_ZERO and abs(Y[p,q]) <= NUMERICAL_ZERO:
                continue
            else:
                Z[p,q] = -Y[p,q] / gap

    X = 0.5 * (Z + Z.T)
    h = nevpt.h1e
    D = np.zeros((nmo,nmo))
    D[c,c] = 2 * np.eye(ncore)
    if nevpt.canonstep == 0 or nevpt.canonstep == 2:
        D[a,a] = nevpt.sa_casdm1
    elif nevpt.canonstep == 1:
        D[a,a] = nevpt.casdm1
    rdm1 = X.copy()
    X_ao = reduce(lib.dot, (nevpt.mo, X, nevpt.mo.T))
    D_ao = reduce(lib.dot, (nevpt.mo, D, nevpt.mo.T))
    vj, vk = nevpt.mf.get_jk(dm=np.array([X_ao, D_ao]))
    veff = vj - 0.5 * vk
    dm1e = reduce(lib.dot, (nevpt.mo.T, veff[0], nevpt.mo))

    if nevpt.canonstep == 1:
        canon_fock = nevpt.fock
    elif nevpt.canonstep == 0 or nevpt.canonstep == 2:
        canon_fock = nevpt.sa_fock
    XF = X @ canon_fock - canon_fock @ X
    natocc = np.diag(nevpt.sa_casdm1)
    nat_response = np.zeros((ncas,ncas))
    for p in range(ncas):
        for q in range(p):
            occ_gap = natocc[p] - natocc[q]
            if abs(occ_gap) < NUMERICAL_ZERO:
                continue
            nat_response[p,q] = XF[ncore+p, ncore+q] / occ_gap
            nat_response[q,p] = nat_response[p,q]

    if nevpt.canonstep == 0 or nevpt.canonstep == 2:
        gci = [2.0 * fci.direct_spin1.contract_1e(dm1e[a,a], ci, nevpt.ncas, nevpt.nelecas) / nevpt.nstates for ci in nevpt.ci]
    elif nevpt.canonstep == 1:
        gci = [np.zeros_like(ci) for ci in nevpt.ci]
        gci[nevpt.state] += 2.0 * fci.direct_spin1.contract_1e(dm1e[a,a], nevpt.state_ci, nevpt.ncas, nevpt.nelecas)
    for root, ci in enumerate(nevpt.ci):
        gci[root] += 2.0 * fci.direct_spin1.contract_1e(nat_response, ci, nevpt.ncas, nevpt.nelecas) / nevpt.nstates
    
    gorb = lib.dot(h, rdm1.T) + lib.dot(h.T, rdm1)
    Imat = 2 * lib.dot(veff[0], D_ao)
    Imat += 2 * lib.dot(veff[1], X_ao)
    gorb += reduce(lib.dot, (nevpt.mo.T, Imat, nevpt.mf.get_ovlp(), nevpt.mo))

    return gorb, gci, rdm1, X, D

# def get_semicanonical_response(nevpt_grad, gorb):
#     nevpt = nevpt_grad.nevpt
#     ncore = nevpt.ncore
#     nocc = nevpt.nocc
#     nmo = nevpt.nmo
#     if nevpt.canonstep == 0 or nevpt.canonstep == 2:
#         mo_energy = np.diag(nevpt.sa_fock)
#     elif nevpt.canonstep == 1:
#         mo_energy = nevpt.mo_energy
#     else:
#         raise NotImplementedError
#     c = nevpt_grad.c
#     a = nevpt_grad.a

#     Y = gorb - gorb.T
#     Z = np.zeros((nmo,nmo))
#     for pi, p in enumerate(range(ncore)):
#         for q in range(pi):
#             gap = mo_energy[p] - mo_energy[q]
#             if abs(gap) < NUMERICAL_ZERO and abs(Y[p,q]) > NUMERICAL_ZERO:
#                 raise RuntimeError('Degeneracy in core-core block')
#             elif abs(gap) < NUMERICAL_ZERO and abs(Y[p,q]) <= NUMERICAL_ZERO:
#                 continue
#             else:
#                 Z[p,q] = -Y[p,q] / gap
#     for pi, p in enumerate(range(nocc, nmo)):
#         for q in range(nocc, nocc+pi):
#             gap = mo_energy[p] - mo_energy[q]
#             if abs(gap) < NUMERICAL_ZERO and abs(Y[p,q]) > NUMERICAL_ZERO:
#                 raise RuntimeError('Degeneracy in extern-extern block')
#             elif abs(gap) < NUMERICAL_ZERO and abs(Y[p,q]) <= NUMERICAL_ZERO:
#                 continue
#             else:
#                 Z[p,q] = -Y[p,q] / gap

#     X = 0.5 * (Z + Z.T)
#     h = nevpt.h1e
#     if hasattr(nevpt.mc, 'with_df'):
#         eri = nevpt.mc.with_df.ao2mo(nevpt.mo, compact=False).reshape([nmo]*4)
#     else:
#         eri = ao2mo.kernel(nevpt.mf._eri, nevpt.mo, compact=False).reshape([nmo]*4)
#         D = np.zeros((nmo,nmo))
#     D[c,c] = 2 * np.eye(ncore)
#     if nevpt.canonstep == 0 or nevpt.canonstep == 2:
#         D[a,a] = nevpt.sa_casdm1
#     elif nevpt.canonstep == 1:
#         D[a,a] = nevpt.casdm1
#     rdm1 = X.copy()
#     rdm2 = einsum('pq,rs->pqrs', X, D)
#     rdm2 -= 0.5 * einsum('pq,rs->prsq', X, D)
#     dm1e = einsum('pq,pqrs->rs', X, eri)
#     dm1e -= 0.5 * einsum('pq,prsq->rs', X, eri)

#     if nevpt.canonstep == 0 or nevpt.canonstep == 2:
#         gci = [2.0 * fci.direct_spin1.contract_1e(dm1e[a,a], ci, nevpt.ncas, nevpt.nelecas) / nevpt.nstates for ci in nevpt.ci]
#     elif nevpt.canonstep == 1:
#         gci = [np.zeros_like(ci) for ci in nevpt.ci]
#         gci[nevpt.state] += 2.0 * fci.direct_spin1.contract_1e(dm1e[a,a], nevpt.state_ci, nevpt.ncas, nevpt.nelecas)
    
#     gorb = lib.dot(h, rdm1.T) + lib.dot(h.T, rdm1)
#     gorb += np.einsum('tqrs,pqrs->tp', eri, rdm2)
#     gorb += np.einsum('ptrs,pqrs->tq', eri, rdm2)
#     gorb += np.einsum('pqts,pqrs->tr', eri, rdm2)
#     gorb += np.einsum('pqrt,pqrs->ts', eri, rdm2)

#     return gorb, gci, rdm1, rdm2