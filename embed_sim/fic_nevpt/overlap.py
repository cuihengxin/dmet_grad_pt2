import numpy as np
from functools import reduce
from numpy import einsum
from functools import partial
einsum = partial(einsum, optimize=True)

def compute_S12_p1(nevpt):
    ncas = nevpt.ncas
    rdms = nevpt.rdms
    rdm_ca = rdms.ca
    s_thresh = nevpt.overlap_threshold

    S_p1  = einsum('XY->XY', np.identity(ncas)).copy()
    S_p1 -= 1/2 * einsum('YX->XY', rdm_ca).copy()
    S_eval, S_evec = np.linalg.eigh(S_p1)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]
    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])
    S_evec = S_evec[:, S_ind_nonzero]
    S_p1_12_inv = np.dot(S_evec, np.diag(S_inv_eval))

    nevpt.log.debug("Dimension of the [+1] orthonormalized subspace:    %d" % S_eval[S_ind_nonzero].shape[0])
    if len(S_ind_nonzero) > 0:
        nevpt.log.debug("Smallest eigenvalue of the [+1] overlap metric:    %e" % np.amin(S_eval[S_ind_nonzero]))
    return S_p1, S_p1_12_inv

def compute_S12_m1(nevpt):
    rdms = nevpt.rdms
    rdm_ca = rdms.ca
    s_thresh = nevpt.overlap_threshold

    S_m1  = 1/2 * einsum('XY->XY', rdm_ca).copy()
    S_eval, S_evec = np.linalg.eigh(S_m1)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]
    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])
    S_evec = S_evec[:, S_ind_nonzero]
    S_m1_12_inv = np.dot(S_evec, np.diag(S_inv_eval))

    nevpt.log.debug("Dimension of the [-1] orthonormalized subspace:    %d" % S_eval[S_ind_nonzero].shape[0])
    if len(S_ind_nonzero) > 0:
        nevpt.log.debug("Smallest eigenvalue of the [-1] overlap metric:    %e" % np.amin(S_eval[S_ind_nonzero]))
    return S_m1, S_m1_12_inv

def compute_S12_p2(nevpt):
    ncas = nevpt.ncas
    rdms = nevpt.rdms
    rdm_ca = rdms.ca
    rdm_ccaa = rdms.ccaa
    s_thresh = nevpt.overlap_threshold

    S_p2  = 1/3 * einsum('WZXY->XYWZ', rdm_ccaa).copy()
    S_p2 += 1/6 * einsum('WZYX->XYWZ', rdm_ccaa).copy()
    S_p2 += einsum('WX,YZ->XYWZ', np.identity(ncas), np.identity(ncas))
    S_p2 -= 1/2 * einsum('WX,YZ->XYWZ', np.identity(ncas), rdm_ca)
    S_p2 -= 1/2 * einsum('YZ,WX->XYWZ', np.identity(ncas), rdm_ca)
    S_p2 = S_p2.reshape(ncas**2, ncas**2)
    S_eval, S_evec = np.linalg.eigh(S_p2)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]
    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])
    S_evec = S_evec[:, S_ind_nonzero]
    S_p2_12_inv = np.dot(S_evec, np.diag(S_inv_eval))

    nevpt.log.debug("Dimension of the [+2] orthonormalized subspace:    %d" % S_eval[S_ind_nonzero].shape[0])
    if len(S_ind_nonzero) > 0:
        nevpt.log.debug("Smallest eigenvalue of the [+2] overlap metric:    %e" % np.amin(S_eval[S_ind_nonzero]))
    return S_p2, S_p2_12_inv

def compute_S12_m2(nevpt):
    ncas = nevpt.ncas
    rdms = nevpt.rdms
    rdm_ccaa = rdms.ccaa
    s_thresh = nevpt.overlap_threshold

    S_m2  = 1/3 * einsum('WZXY->XYWZ', rdm_ccaa).copy()
    S_m2 += 1/6 * einsum('WZYX->XYWZ', rdm_ccaa).copy()
    S_m2 = S_m2.reshape(ncas**2, ncas**2)
    S_eval, S_evec = np.linalg.eigh(S_m2)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]
    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])
    S_evec = S_evec[:, S_ind_nonzero]
    S_m2_12_inv = np.dot(S_evec, np.diag(S_inv_eval))

    nevpt.log.debug("Dimension of the [-2] orthonormalized subspace:    %d" % S_eval[S_ind_nonzero].shape[0])
    if len(S_ind_nonzero) > 0:
        nevpt.log.debug("Smallest eigenvalue of the [-2] overlap metric:    %e" % np.amin(S_eval[S_ind_nonzero]))
    return S_m2, S_m2_12_inv

def compute_S12_0p(nevpt):
    ncas = nevpt.ncas
    rdms = nevpt.rdms
    rdm_ca = rdms.ca
    rdm_ccaa = rdms.ccaa
    s_thresh = nevpt.overlap_threshold

    S22_aa_aa =- 1/6 * einsum('WYXZ->XYWZ', rdm_ccaa).copy()
    S22_aa_aa += 1/6 * einsum('WYZX->XYWZ', rdm_ccaa).copy()
    S22_aa_aa += 1/2 * einsum('YZ,XW->XYWZ', np.identity(ncas), rdm_ca)
    S22_aa_bb  = 1/6 * einsum('WYXZ->XYWZ', rdm_ccaa).copy()
    S22_aa_bb += 1/3 * einsum('WYZX->XYWZ', rdm_ccaa).copy()

    dim_wz = ncas * ncas
    dim_S_0p = 2 * dim_wz
    S22_aa_aa = S22_aa_aa.reshape(dim_wz, dim_wz)
    S22_aa_bb = S22_aa_bb.reshape(dim_wz, dim_wz)

    s_aa = 0
    f_aa = s_aa + dim_wz
    s_bb = f_aa
    f_bb = s_bb + dim_wz
    S_0p = np.zeros((dim_S_0p, dim_S_0p))
    S_0p[s_aa:f_aa, s_aa:f_aa] = S22_aa_aa.copy()
    S_0p[s_bb:f_bb, s_bb:f_bb] = S22_aa_aa.copy()
    S_0p[s_aa:f_aa, s_bb:f_bb] = S22_aa_bb.copy()
    S_0p[s_bb:f_bb, s_aa:f_aa] = S22_aa_bb.T.copy()
    S_eval, S_evec = np.linalg.eigh(S_0p)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]
    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])
    S_evec = S_evec[:, S_ind_nonzero]
    S_0p_12_inv = reduce(np.dot, (S_evec, np.diag(S_inv_eval)))

    nevpt.log.debug("Dimension of the [0'] orthonormalized subspace:    %d" % S_eval[S_ind_nonzero].shape[0])
    if len(S_ind_nonzero) > 0:
        nevpt.log.debug("Smallest eigenvalue of the [0'] overlap metric:    %e" % np.amin(S_eval[S_ind_nonzero]))
    return S_0p, S_0p_12_inv

def compute_S12_p1p(nevpt):
    ncas = nevpt.ncas
    rdms = nevpt.rdms
    rdm_ca = rdms.ca
    rdm_ccaa = rdms.ccaa
    rdm_cccaaa = rdms.cccaaa
    s_thresh = nevpt.overlap_threshold

    S22_aaa_aaa =- 1/12 * einsum('UYZVXW->UVXWZY', rdm_cccaaa).copy()
    S22_aaa_aaa -= 1/12 * einsum('UYZWVX->UVXWZY', rdm_cccaaa).copy()
    S22_aaa_aaa -= 1/12 * einsum('UYZXWV->UVXWZY', rdm_cccaaa).copy()
    S22_aaa_aaa += 1/6 * einsum('VY,UZWX->UVXWZY', np.identity(ncas), rdm_ccaa)
    S22_aaa_aaa -= 1/6 * einsum('VY,UZXW->UVXWZY', np.identity(ncas), rdm_ccaa)
    S22_aaa_aaa -= 1/6 * einsum('VZ,UYWX->UVXWZY', np.identity(ncas), rdm_ccaa)
    S22_aaa_aaa += 1/6 * einsum('VZ,UYXW->UVXWZY', np.identity(ncas), rdm_ccaa)
    S22_aaa_aaa += 1/6 * einsum('XY,UZVW->UVXWZY', np.identity(ncas), rdm_ccaa)
    S22_aaa_aaa -= 1/6 * einsum('XY,UZWV->UVXWZY', np.identity(ncas), rdm_ccaa)
    S22_aaa_aaa -= 1/6 * einsum('XZ,UYVW->UVXWZY', np.identity(ncas), rdm_ccaa)
    S22_aaa_aaa += 1/6 * einsum('XZ,UYWV->UVXWZY', np.identity(ncas), rdm_ccaa)
    S22_aaa_aaa -= 1/2 * einsum('VY,XZ,UW->UVXWZY', np.identity(ncas), np.identity(ncas), rdm_ca)
    S22_aaa_aaa += 1/2 * einsum('VZ,XY,UW->UVXWZY', np.identity(ncas), np.identity(ncas), rdm_ca)
    S22_bba_bba =- 1/12 * einsum('UYZVXW->UVXWZY', rdm_cccaaa).copy()
    S22_bba_bba += 1/12 * einsum('UYZWVX->UVXWZY', rdm_cccaaa).copy()
    S22_bba_bba += 1/6 * einsum('UYZWXV->UVXWZY', rdm_cccaaa).copy()
    S22_bba_bba += 1/12 * einsum('UYZXWV->UVXWZY', rdm_cccaaa).copy()
    S22_bba_bba -= 1/3 * einsum('VZ,UYWX->UVXWZY', np.identity(ncas), rdm_ccaa)
    S22_bba_bba -= 1/6 * einsum('VZ,UYXW->UVXWZY', np.identity(ncas), rdm_ccaa)
    S22_bba_bba += 1/6 * einsum('XY,UZVW->UVXWZY', np.identity(ncas), rdm_ccaa)
    S22_bba_bba -= 1/6 * einsum('XY,UZWV->UVXWZY', np.identity(ncas), rdm_ccaa)
    S22_bba_bba += 1/2 * einsum('VZ,XY,UW->UVXWZY', np.identity(ncas), np.identity(ncas), rdm_ca)
    S22_aaa_bba = np.ascontiguousarray(S22_aaa_aaa -
                                       S22_bba_bba.transpose(0,2,1,3,5,4) +
                                       S22_bba_bba.transpose(0,1,2,3,5,4))
    S22_bba_aaa = np.ascontiguousarray(S22_aaa_bba.transpose(3,4,5,0,1,2))

    dim_wzy = ncas * ncas * ncas
    dim_tril_wzy = ncas * ncas * (ncas - 1) // 2
    dim_act = dim_wzy + dim_tril_wzy
    tril_ind = np.tril_indices(ncas, k=-1)
    S22_aaa_aaa = S22_aaa_aaa[:, :, :, :, tril_ind[0], tril_ind[1]]
    S22_aaa_aaa = S22_aaa_aaa[:, tril_ind[0], tril_ind[1]]
    S22_aaa_bba = S22_aaa_bba[:, tril_ind[0], tril_ind[1]]
    S22_bba_aaa = S22_bba_aaa[:, :, :, :, tril_ind[0], tril_ind[1]]
    S22_aaa_aaa = S22_aaa_aaa.reshape(dim_tril_wzy, dim_tril_wzy)
    S22_aaa_bba = S22_aaa_bba.reshape(dim_tril_wzy, dim_wzy)
    S22_bba_aaa = S22_bba_aaa.reshape(dim_wzy, dim_tril_wzy)
    S22_bba_bba = S22_bba_bba.reshape(dim_wzy, dim_wzy)

    s_aaa = 0
    f_aaa = s_aaa + dim_tril_wzy
    s_bba = f_aaa
    f_bba = s_bba + dim_wzy
    S_p1p_act = np.zeros((dim_act, dim_act))
    S_p1p_act[s_aaa:f_aaa, s_aaa:f_aaa] = S22_aaa_aaa
    S_p1p_act[s_aaa:f_aaa, s_bba:f_bba] = S22_aaa_bba
    S_p1p_act[s_bba:f_bba, s_aaa:f_aaa] = S22_bba_aaa
    S_p1p_act[s_bba:f_bba, s_bba:f_bba] = S22_bba_bba

    S_eval, S_evec = np.linalg.eigh(S_p1p_act)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]
    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])
    S_evec = S_evec[:, S_ind_nonzero]
    S_p1p_12_inv_act = reduce(np.dot, (S_evec, np.diag(S_inv_eval)))

    nevpt.log.debug("Dimension of the [+1'] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
    if len(S_ind_nonzero) > 0:
        nevpt.log.debug("Smallest eigenvalue of the [+1'] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))
    return S_p1p_act, S_p1p_12_inv_act

def compute_S12_m1p(nevpt):
    ncas = nevpt.ncas
    rdms = nevpt.rdms
    rdm_ca = rdms.ca
    rdm_ccaa = rdms.ccaa
    rdm_cccaaa = rdms.cccaaa
    s_thresh = nevpt.overlap_threshold

    S22_aaa_aaa =- 1/12 * einsum('UVYWZX->UVXWZY', rdm_cccaaa).copy()
    S22_aaa_aaa -= 1/12 * einsum('UVYXWZ->UVXWZY', rdm_cccaaa).copy()
    S22_aaa_aaa -= 1/12 * einsum('UVYZXW->UVXWZY', rdm_cccaaa).copy()
    S22_aaa_aaa += 1/6 * einsum('XY,UVWZ->UVXWZY', np.identity(ncas), rdm_ccaa)
    S22_aaa_aaa -= 1/6 * einsum('XY,UVZW->UVXWZY', np.identity(ncas), rdm_ccaa)
    S22_abb_abb  = 1/6 * einsum('UVYWXZ->UVXWZY', rdm_cccaaa).copy()
    S22_abb_abb -= 1/12 * einsum('UVYWZX->UVXWZY', rdm_cccaaa).copy()
    S22_abb_abb += 1/12 * einsum('UVYXWZ->UVXWZY', rdm_cccaaa).copy()
    S22_abb_abb += 1/12 * einsum('UVYZXW->UVXWZY', rdm_cccaaa).copy()
    S22_abb_abb += 1/3 * einsum('XY,UVWZ->UVXWZY', np.identity(ncas), rdm_ccaa)
    S22_abb_abb += 1/6 * einsum('XY,UVZW->UVXWZY', np.identity(ncas), rdm_ccaa)
    S22_aaa_abb = np.ascontiguousarray(S22_aaa_aaa -
                                       S22_abb_abb.transpose(1,0,2,4,3,5) +
                                       S22_abb_abb.transpose(0,1,2,4,3,5))
    S22_abb_aaa = np.ascontiguousarray(S22_aaa_abb.transpose(3,4,5,0,1,2))

    dim_wzy = ncas * ncas * ncas
    dim_tril_wzy = ncas * ncas * (ncas - 1) // 2
    dim_act = dim_wzy + dim_tril_wzy
    tril_ind = np.tril_indices(ncas, k=-1)
    S22_aaa_aaa = S22_aaa_aaa[:, :, :, tril_ind[0], tril_ind[1]]
    S22_aaa_aaa = S22_aaa_aaa[tril_ind[0], tril_ind[1]]
    S22_aaa_abb = S22_aaa_abb[tril_ind[0], tril_ind[1]]
    S22_abb_aaa = S22_abb_aaa[:, :, :, tril_ind[0], tril_ind[1]]
    S22_aaa_aaa = S22_aaa_aaa.reshape(dim_tril_wzy, dim_tril_wzy)
    S22_aaa_abb = S22_aaa_abb.reshape(dim_tril_wzy, dim_wzy)
    S22_abb_aaa = S22_abb_aaa.reshape(dim_wzy, dim_tril_wzy)
    S22_abb_abb = S22_abb_abb.reshape(dim_wzy, dim_wzy)

    s_aaa = 0
    f_aaa = s_aaa + dim_tril_wzy
    s_abb = f_aaa
    f_abb = s_abb + dim_wzy
    S_m1p_act = np.zeros((dim_act, dim_act))
    S_m1p_act[s_aaa:f_aaa, s_aaa:f_aaa] = S22_aaa_aaa
    S_m1p_act[s_aaa:f_aaa, s_abb:f_abb] = S22_aaa_abb
    S_m1p_act[s_abb:f_abb, s_aaa:f_aaa] = S22_abb_aaa
    S_m1p_act[s_abb:f_abb, s_abb:f_abb] = S22_abb_abb

    S_eval, S_evec = np.linalg.eigh(S_m1p_act)
    S_ind_nonzero = np.where(S_eval > s_thresh)[0]
    S_inv_eval = 1.0/np.sqrt(S_eval[S_ind_nonzero])
    S_evec = S_evec[:, S_ind_nonzero]
    S_m1p_12_inv_act = reduce(np.dot, (S_evec, np.diag(S_inv_eval)))

    nevpt.log.debug("Dimension of the [-1'] orthonormalized subspace:   %d" % S_eval[S_ind_nonzero].shape[0])
    if len(S_ind_nonzero) > 0:
        nevpt.log.debug("Smallest eigenvalue of the [-1'] overlap metric:   %e" % np.amin(S_eval[S_ind_nonzero]))
    return S_m1p_act, S_m1p_12_inv_act