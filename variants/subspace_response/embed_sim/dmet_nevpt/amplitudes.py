from . import intermediates
from . import overlap
from pyscf import lib
from pyscf.lib import logger
import numpy as np
from numpy import einsum
from functools import reduce
from functools import partial
einsum = partial(einsum, optimize=True)

def solve_amplitude(koopmans, overlap_discard, V1, Delta):
    SKS = reduce(lib.dot, (overlap_discard.T, koopmans, overlap_discard))
    SKS = np.tril(SKS) + np.tril(SKS, k=-1).T
    A = SKS + Delta[...,None,None] * np.eye(overlap_discard.shape[-1])
    B = einsum('...P,Pm->...m', V1, overlap_discard)
    ic_coeff = np.linalg.solve(A, B[...,None])[...,0]
    return ic_coeff

def compute_t1_0(nevpt):
    cput0 = (logger.process_clock(), logger.perf_counter())
    nevpt.log.debug("\nComputing T[0]^(1) amplitudes...")

    ncore = nevpt.ncore
    nextern = nevpt.nextern
    e_core = nevpt.mo_energy.c
    e_extern = nevpt.mo_energy.e
    v_cece = nevpt.v2e.cece

    d_ij = e_core[:,None] + e_core
    d_ab = e_extern[:,None] + e_extern
    V1_0 = - einsum('IAJB->IJAB', v_cece)
    t1_ccee = -d_ij.reshape(-1,1) + d_ab.reshape(-1)
    t1_ccee = t1_ccee.reshape((ncore, ncore, -1, nextern))
    t1_ccee = t1_ccee**(-1)
    t1_ccee *= V1_0
    # e_0 = 2 * einsum('ijab,iajb', t1_ccee, v_cece)
    # e_0 -= einsum('ijab,jaib', t1_ccee, v_cece)
    t1_ccee_ = 2 * einsum('iajb->ijab', v_cece)
    t1_ccee_ -= einsum('jaib->ijab', v_cece)
    e_0 = einsum('ijab,ijab', t1_ccee, t1_ccee_)

    nevpt.log.debug("Norm of T[0]^(1):                            %20.12f" % np.linalg.norm(t1_ccee))
    nevpt.log.info("Correlation energy [0]:                      %20.12f" % e_0)
    nevpt.log.timer("computing T[0]^(1) amplitudes", *cput0)

    if e_0 > 0.0:
        nevpt.log.info("\n\nWARNING: Positive correlation energy detected in class [0] !!!!\n\n")
    return e_0, t1_ccee, None, None, None, V1_0, None, t1_ccee_

def compute_t1_p1(nevpt):
    cput0 = (logger.process_clock(), logger.perf_counter())
    nevpt.log.debug("\nComputing T[+1]^(1) amplitudes...")

    e_core = nevpt.mo_energy.c
    e_extern = nevpt.mo_energy.e
    v_cace = nevpt.v2e.cace
    rdms = nevpt.rdms
    rdm_ca = rdms.ca

    K_ac = intermediates.compute_K_ac(nevpt)
    S_p1, S_p1_12_inv_act = overlap.compute_S12_p1(nevpt)
    # SKS = reduce(np.dot, (S_p1_12_inv_act.T, K_ac, S_p1_12_inv_act))
    # evals, evecs = np.linalg.eigh(SKS)
    # del(SKS)
    V1_p1 =- einsum('JXIA->IJAX', v_cace).copy()
    V1_p1 += 1/2 * einsum('JxIA,Xx->IJAX', v_cace, rdm_ca)
    # d_ap = (e_extern[:,None] + evals).reshape(-1)
    # d_ij = (e_core[:,None] + e_core).reshape(-1)
    # d_apij = (d_ap[:,None] - d_ij).reshape(nextern, evals.shape[0], ncore, ncore)
    # d_apij = d_apij**(-1)

    # S_12_V_p1 = einsum("IJAX,Xm->IJAm", V1_p1, S_p1_12_inv_act)
    # S_12_V_p1 = einsum("mp,IJAm->IJAp", evecs, S_12_V_p1)
    # S_12_V_p1 = einsum("ApIJ,IJAp->IJAp", d_apij, S_12_V_p1)
    # S_12_V_p1 = einsum("mp,IJAp->IJAm", evecs, S_12_V_p1)
    # del(d_ap, d_ij, d_apij, evals, evecs)
    # t1_ccae = einsum("IJAm,Xm->JIXA", S_12_V_p1, S_p1_12_inv_act).copy()
    # del(S_12_V_p1)
    # e_p1 =- 2 * einsum('ijxa,jxia', t1_ccae, v_cace)
    # e_p1 += 4 * einsum('ijxa,ixja', t1_ccae, v_cace)
    # e_p1 += einsum('ijxa,jyia,xy', t1_ccae, v_cace, rdm_ca)
    # e_p1 -= 2 * einsum('ijxa,iyja,xy', t1_ccae, v_cace, rdm_ca)

    Delta = e_extern[None, None, :] - e_core[:, None, None] - e_core[None, :, None]
    ic_coeff = solve_amplitude(K_ac, S_p1_12_inv_act, V1_p1, Delta)
    t1_ccae = einsum('...m,Pm->...P', ic_coeff, S_p1_12_inv_act).transpose(1,0,3,2)
    t1_ccae_ =- 2 * einsum('jxia->ijxa', v_cace)
    t1_ccae_ += 4 * einsum('ixja->ijxa', v_cace)
    t1_ccae_ += einsum('jyia,xy->ijxa', v_cace, rdm_ca)
    t1_ccae_ -= 2 * einsum('iyja,xy->ijxa', v_cace, rdm_ca)
    e_p1 = einsum('ijxa,ijxa', t1_ccae, t1_ccae_)

    nevpt.log.debug("Norm of T[+1]^(1):                           %20.12f" % np.linalg.norm(t1_ccae))
    nevpt.log.info("Correlation energy [+1]:                     %20.12f" % e_p1)
    nevpt.log.timer("computing T[+1]^(1) amplitudes", *cput0)

    if e_p1 > 0.0:
        nevpt.log.info("\n\nWARNING: Positive correlation energy detected in class [+1] !!!!\n\n")
    return e_p1, t1_ccae, K_ac, S_p1, S_p1_12_inv_act, V1_p1, ic_coeff, t1_ccae_

def compute_t1_m1(nevpt):
    cput0 = (logger.process_clock(), logger.perf_counter())
    nevpt.log.debug("\nComputing T[-1]^(1) amplitudes...")

    e_core = nevpt.mo_energy.c
    e_extern = nevpt.mo_energy.e
    v_ceae = nevpt.v2e.ceae
    rdms = nevpt.rdms
    rdm_ca = rdms.ca

    K_ca = intermediates.compute_K_ca(nevpt)
    S_m1, S_m1_12_inv_act = overlap.compute_S12_m1(nevpt)
    # SKS = reduce(np.dot, (S_m1_12_inv_act.T, K_ca, S_m1_12_inv_act))
    # evals, evecs = np.linalg.eigh(SKS)
    # del(SKS)
    # d_ix = (e_core[:,None] - evals).reshape(-1)
    # t1_caee = np.zeros((ncore, ncas, nextern, nextern))
    # d_ab = (e_extern[:,None] + e_extern).reshape(-1)
    # d_abix = (d_ab[:,None] - d_ix).reshape(-1, nextern, ncore, evals.shape[0])
    # d_abix = d_abix**(-1)
    V1_m1 = -1/2 * einsum('IAxB,Xx->IXAB', v_ceae, rdm_ca).transpose(0,2,3,1)
    # S_12_V_m1 = einsum("IXAB,Xm->ImAB", V1_m1, S_m1_12_inv_act)
    # S_12_V_m1 = einsum("mp,ImAB->IpAB", evecs, S_12_V_m1)
    # S_12_V_m1 = einsum("ABIp,IpAB->IpAB", d_abix, S_12_V_m1)
    # S_12_V_m1 = einsum("mp,IpAB->ImAB", evecs, S_12_V_m1)
    # del(d_abix)
    # temp = einsum("ImAB,Xm->IXAB", S_12_V_m1, S_m1_12_inv_act).copy()
    # del(S_12_V_m1)
    # e_m1 = 2 * einsum('ixab,iayb,xy', temp, v_ceae, rdm_ca)
    # e_m1 -= einsum('ixab,ibya,xy', temp, v_ceae, rdm_ca)
    # t1_caee = temp

    Delta = e_extern[None, None, :] + e_extern[None, :, None] - e_core[:, None, None]
    ic_coeff = solve_amplitude(K_ca, S_m1_12_inv_act, V1_m1, Delta)
    t1_caee = einsum('...m,Pm->...P', ic_coeff, S_m1_12_inv_act).transpose(0,3,1,2)
    t1_caee_ = 2 * einsum('iayb,xy->ixab', v_ceae, rdm_ca)
    t1_caee_ -= einsum('ibya,xy->ixab', v_ceae, rdm_ca)
    e_m1 = einsum('ixab,ixab', t1_caee, t1_caee_)

    nevpt.log.debug("Norm of T[-1]^(1):                           %20.12f" % np.linalg.norm(t1_caee))
    nevpt.log.info("Correlation energy [-1]:                     %20.12f" % e_m1)
    nevpt.log.timer("computing T[-1]^(1) amplitudes", *cput0)

    if e_m1 > 0.0:
        nevpt.log.info("\n\nWARNING: Positive correlation energy detected in class [-1] !!!!\n\n")
    return e_m1, t1_caee, K_ca, S_m1, S_m1_12_inv_act, V1_m1, ic_coeff, t1_caee_

def compute_t1_p2(nevpt):
    cput0 = (logger.process_clock(), logger.perf_counter())
    nevpt.log.debug("\nComputing T[+2]^(1) amplitudes...")

    ncore = nevpt.ncore
    ncas = nevpt.ncas
    e_core = nevpt.mo_energy.c
    v_caca = nevpt.v2e.caca
    rdms = nevpt.rdms
    rdm_ca = rdms.ca
    rdm_ccaa = rdms.ccaa

    K_aacc = intermediates.compute_K_aacc(nevpt)
    S_p2, S_p2_12_inv_act = overlap.compute_S12_p2(nevpt)
    # SKS = reduce(np.dot, (S_p2_12_inv_act.T, K_aacc, S_p2_12_inv_act))
    # evals, evecs = np.linalg.eigh(SKS)
    # del(SKS)
    V1_p2 =- einsum('IXJY->IJXY', v_caca).copy()
    V1_p2 += 1/2 * einsum('IXJx,Yx->IJXY', v_caca, rdm_ca)
    V1_p2 += 1/2 * einsum('IxJY,Xx->IJXY', v_caca, rdm_ca)
    V1_p2 -= 1/3 * einsum('IxJy,XYxy->IJXY', v_caca, rdm_ccaa)
    V1_p2 -= 1/6 * einsum('IxJy,XYyx->IJXY', v_caca, rdm_ccaa)
    V1_p2 = V1_p2.reshape(ncore, ncore, ncas**2)
    # d_ij = (e_core[:,None] + e_core).reshape(-1)
    # d_pij = (evals[:,None] - d_ij).reshape(evals.shape[0], ncore, ncore)
    # d_pij = d_pij**(-1)
    # S_12_V_p2 = einsum("IJX,Xm->IJm", V1_p2, S_p2_12_inv_act)
    # S_12_V_p2 = einsum("mp,IJm->IJp", evecs, S_12_V_p2)
    # S_12_V_p2 = einsum("pIJ,IJp->IJp", d_pij, S_12_V_p2)
    # S_12_V_p2 = einsum("mp,IJp->IJm", evecs, S_12_V_p2)
    # del(d_ij, d_pij)
    # t1_ccaa = einsum("IJm,Xm->IJX", S_12_V_p2, S_p2_12_inv_act)
    # t1_ccaa = t1_ccaa.reshape(ncore, ncore, ncas, ncas)
    # del(S_12_V_p2)
    # e_p2  = 2 * einsum('ijxy,ixjy', t1_ccaa, v_caca)
    # e_p2 -= einsum('ijxy,jxiy', t1_ccaa, v_caca)
    # e_p2 += 1/2 * einsum('ijxy,izjw,xyzw', t1_ccaa, v_caca, rdm_ccaa)
    # e_p2 += einsum('ijxy,iyjz,xz', t1_ccaa, v_caca, rdm_ca)
    # e_p2 -= 2 * einsum('ijxy,ixjz,yz', t1_ccaa, v_caca, rdm_ca)

    Delta = -e_core[:, None] - e_core[None, :]
    ic_coeff = solve_amplitude(K_aacc, S_p2_12_inv_act, V1_p2, Delta)
    t1_ccaa = einsum('...m,Pm->...P', ic_coeff, S_p2_12_inv_act).reshape(ncore,ncore,ncas,ncas)
    t1_ccaa_  = 2 * einsum('ixjy->ijxy', v_caca)
    t1_ccaa_ -= einsum('jxiy->ijxy', v_caca)
    t1_ccaa_ += 1/2 * einsum('izjw,xyzw->ijxy', v_caca, rdm_ccaa)
    t1_ccaa_ += einsum('iyjz,xz->ijxy', v_caca, rdm_ca)
    t1_ccaa_ -= 2 * einsum('ixjz,yz->ijxy', v_caca, rdm_ca)
    e_p2 = einsum('ixab,ixab', t1_ccaa, t1_ccaa_)

    nevpt.log.debug("Norm of T[+2]^(1):                           %20.12f" % np.linalg.norm(t1_ccaa))
    nevpt.log.info("Correlation energy [+2]:                     %20.12f" % e_p2)
    nevpt.log.timer("computing T[+2]^(1) amplitudes", *cput0)

    if e_p2 > 0.0:
        nevpt.log.info("\n\nWARNING: Positive correlation energy detected in class [+2] !!!!\n\n")
    return e_p2, t1_ccaa, K_aacc, S_p2, S_p2_12_inv_act, V1_p2, ic_coeff, t1_ccaa_

def compute_t1_m2(nevpt):
    cput0 = (logger.process_clock(), logger.perf_counter())
    nevpt.log.debug("\nComputing T[-2]^(1) amplitudes...")

    ncas = nevpt.ncas
    nextern = nevpt.nextern
    e_extern = nevpt.mo_energy.e
    v_aeae = nevpt.v2e.aeae
    rdms = nevpt.rdms
    rdm_ccaa = rdms.ccaa

    K_ccaa = intermediates.compute_K_ccaa(nevpt)
    S_m2, S_m2_12_inv_act = overlap.compute_S12_m2(nevpt)
    # SKS = reduce(np.dot, (S_m2_12_inv_act.T, K_ccaa, S_m2_12_inv_act))
    # evals, evecs = np.linalg.eigh(SKS)
    # del(SKS)
    # t1_aaee = np.zeros((ncas, ncas, nextern, nextern))

    # d_ab = (e_extern[:,None] + e_extern).reshape(-1)
    # d_abp = (d_ab[:,None] + evals).reshape(-1, nextern, evals.shape[0])
    # d_abp = d_abp**(-1)
    V1_m2 =- 1/3 * einsum('xAyB,XYxy->XYAB', v_aeae, rdm_ccaa)
    V1_m2 -= 1/6 * einsum('xAyB,XYyx->XYAB', v_aeae, rdm_ccaa)
    V1_m2 = V1_m2.reshape(ncas**2, -1, nextern).transpose(1,2,0)
    # S_12_V_m2 = einsum("XAB,Xm->mAB", V1_m2, S_m2_12_inv_act)
    # S_12_V_m2 = einsum("mp,mAB->pAB", evecs, S_12_V_m2)
    # S_12_V_m2 = einsum("ABp,pAB->pAB", d_abp, S_12_V_m2)
    # S_12_V_m2 = einsum("mp,pAB->mAB", evecs, S_12_V_m2)
    # temp = einsum("mAB,Xm->XAB", S_12_V_m2, S_m2_12_inv_act)
    # temp = temp.reshape(ncas, ncas, -1, nextern)
    # del(S_12_V_m2)
    # e_m2 = 1/2 * einsum('xyab,zawb,xyzw', temp, v_aeae, rdm_ccaa)
    # t1_aaee = temp

    Delta = e_extern[:,None] + e_extern[None,:]
    ic_coeff = solve_amplitude(K_ccaa, S_m2_12_inv_act, V1_m2, Delta)
    t1_aaee = einsum('...m,Pm->...P', ic_coeff, S_m2_12_inv_act).transpose(2,0,1).reshape(ncas,ncas,nextern,nextern)
    t1_aaee_  = 1/2 * einsum('zawb,xyzw->xyab', v_aeae, rdm_ccaa)
    e_m2 = einsum('xyab,xyab', t1_aaee, t1_aaee_)

    nevpt.log.debug("Norm of T[-2]^(1):                           %20.12f" % np.linalg.norm(t1_aaee))
    nevpt.log.info("Correlation energy [-2]:                     %20.12f" % e_m2)
    nevpt.log.timer("computing T[-2]^(1) amplitudes", *cput0)

    if e_m2 > 0.0:
        nevpt.log.info("\n\nWARNING: Positive correlation energy detected in class [-2] !!!!\n\n")
    return e_m2, t1_aaee, K_ccaa, S_m2, S_m2_12_inv_act, V1_m2, ic_coeff, t1_aaee_

def compute_t1_0p(nevpt):
    cput0 = (logger.process_clock(), logger.perf_counter())
    nevpt.log.debug("\nComputing T[0']^(1) amplitudes...")

    ncore = nevpt.ncore
    ncas = nevpt.ncas
    nextern = nevpt.nextern
    e_core = nevpt.mo_energy.c
    e_extern = nevpt.mo_energy.e
    h_ce = nevpt.h1eff.ce
    v_caae = nevpt.v2e.caae
    v_ceaa = nevpt.v2e.ceaa
    rdms = nevpt.rdms
    rdm_ca = rdms.ca
    rdm_ccaa = rdms.ccaa

    K_caca = intermediates.compute_K_caca(nevpt)
    S_0p_act, S_0p_12_inv_act = overlap.compute_S12_0p(nevpt)
    # SKS = reduce(np.dot, (S_0p_12_inv_act.T, K_caca, S_0p_12_inv_act))
    # evals, evecs = np.linalg.eigh(SKS)
    # del(SKS)
    V2_aa_aa =- 1/2 * einsum('IA,XY->IAXY', h_ce, rdm_ca)
    V2_aa_aa -= 1/2 * einsum('IAxY,Xx->IAXY', v_ceaa, rdm_ca)
    V2_aa_aa -= 1/2 * einsum('IAxy,XyYx->IAXY', v_ceaa, rdm_ccaa)
    V2_aa_aa += 1/2 * einsum('IYxA,Xx->IAXY', v_caae, rdm_ca)
    V2_aa_aa += 1/6 * einsum('IxyA,XxYy->IAXY', v_caae, rdm_ccaa)
    V2_aa_aa -= 1/6 * einsum('IxyA,XxyY->IAXY', v_caae, rdm_ccaa)
    V2_aa_bb =- 1/2 * einsum('IA,XY->IAXY', h_ce, rdm_ca)
    V2_aa_bb -= 1/2 * einsum('IAxY,Xx->IAXY', v_ceaa, rdm_ca)
    V2_aa_bb -= 1/2 * einsum('IAxy,XyYx->IAXY', v_ceaa, rdm_ccaa)
    V2_aa_bb += 1/3 * einsum('IxyA,XxYy->IAXY', v_caae, rdm_ccaa)
    V2_aa_bb += 1/6 * einsum('IxyA,XxyY->IAXY', v_caae, rdm_ccaa)
    V2_aa_aa = V2_aa_aa.reshape(ncore, nextern, -1)
    V2_aa_bb = V2_aa_bb.reshape(ncore, nextern, -1)

    dim_XY = ncas * ncas
    dim_act = 2 * dim_XY
    V_aa_aa_i = 0
    V_aa_aa_f = V_aa_aa_i + dim_XY
    V_aa_bb_i = V_aa_aa_f
    V_aa_bb_f = V_aa_bb_i + dim_XY
    V_0p = np.zeros((ncore, nextern, dim_act))
    V_0p[:,:,V_aa_aa_i:V_aa_aa_f] = V2_aa_aa.copy()
    V_0p[:,:,V_aa_bb_i:V_aa_bb_f] = V2_aa_bb.copy()
    del(V2_aa_aa, V2_aa_bb)
    # d_ai = (e_extern[:,None] - e_core).reshape(-1)
    # d_aip = (d_ai[:,None] + evals).reshape(nextern, ncore, -1)
    # d_aip = d_aip**(-1)
    # S_12_V_0p = einsum("iaP,Pm->iam", V_0p, S_0p_12_inv_act)
    # S_12_V_0p = einsum("mp,iam->iap", evecs, S_12_V_0p)
    # S_12_V_0p = einsum("aip,iap->iap", d_aip, S_12_V_0p)
    # S_12_V_0p = einsum("mp,iap->iam", evecs, S_12_V_0p)
    # del(d_ai, d_aip)
    # t_0p = einsum("iam,Pm->iaP", S_12_V_0p, S_0p_12_inv_act)
    # del(S_12_V_0p)
    # t1_caea_aaaa = t_0p[:,:,V_aa_aa_i:V_aa_aa_f].reshape(ncore, nextern, ncas, ncas).transpose(0,2,1,3)
    # t1_caea_abab = t_0p[:,:,V_aa_bb_i:V_aa_bb_f].reshape(ncore, nextern, ncas, ncas).transpose(0,2,1,3)
    # t1_caea = t1_caea_abab
    # t1_caae = (t1_caea_abab - t1_caea_aaaa).transpose(0,1,3,2).copy()
    # del(t_0p, t1_caea_aaaa, t1_caea_abab)
    # e_0p = 2 * einsum('ia,ixay,yx', h_ce, t1_caea, rdm_ca)
    # e_0p -= einsum('ia,ixya,yx', h_ce, t1_caae, rdm_ca)
    # e_0p += 2 * einsum('ixay,iazw,yzxw', t1_caea, v_ceaa, rdm_ccaa)
    # e_0p += 2 * einsum('ixay,iazy,xz', t1_caea, v_ceaa, rdm_ca)
    # e_0p -= einsum('ixay,izwa,ywxz', t1_caea, v_caae, rdm_ccaa)
    # e_0p -= einsum('ixay,iyza,xz', t1_caea, v_caae, rdm_ca)
    # e_0p -= einsum('ixya,iazw,yzxw', t1_caae, v_ceaa, rdm_ccaa)
    # e_0p -= einsum('ixya,iazy,xz', t1_caae, v_ceaa, rdm_ca)
    # e_0p -= einsum('ixya,izwa,ywzx', t1_caae, v_caae, rdm_ccaa)
    # e_0p += 2 * einsum('ixya,iyza,xz', t1_caae, v_caae, rdm_ca)

    Delta = e_extern[None,:] - e_core[:,None]
    ic_coeff = solve_amplitude(K_caca, S_0p_12_inv_act, V_0p, Delta)
    t_0p = einsum('...m,Pm->...P', ic_coeff, S_0p_12_inv_act)
    t1_caea_aaaa = t_0p[:,:,V_aa_aa_i:V_aa_aa_f].reshape(ncore, nextern, ncas, ncas).transpose(0,2,1,3)
    t1_caea_abab = t_0p[:,:,V_aa_bb_i:V_aa_bb_f].reshape(ncore, nextern, ncas, ncas).transpose(0,2,1,3)
    t1_caea = t1_caea_abab
    t1_caae = (t1_caea_abab - t1_caea_aaaa).transpose(0,1,3,2).copy()
    del(t_0p)
    t1_caea_ = 2 * einsum('ia,yx->ixay', h_ce, rdm_ca)
    t1_caea_ += 2 * einsum('iazw,yzxw->ixay', v_ceaa, rdm_ccaa)
    t1_caea_ += 2 * einsum('iazy,xz->ixay', v_ceaa, rdm_ca)
    t1_caea_ -= einsum('izwa,ywxz->ixay', v_caae, rdm_ccaa)
    t1_caea_ -= einsum('iyza,xz->ixay', v_caae, rdm_ca)
    t1_caae_ = -einsum('ia,yx->ixya', h_ce, rdm_ca)
    t1_caae_ -= einsum('iazw,yzxw->ixya', v_ceaa, rdm_ccaa)
    t1_caae_ -= einsum('iazy,xz->ixya', v_ceaa, rdm_ca)
    t1_caae_ -= einsum('izwa,ywzx->ixya', v_caae, rdm_ccaa)
    t1_caae_ += 2 * einsum('iyza,xz->ixya', v_caae, rdm_ca)
    e_0p = einsum('ixay,ixay',t1_caea, t1_caea_) + einsum('ixya,ixya', t1_caae, t1_caae_)

    nevpt.log.debug("Norm of T[0']^(1):                           %20.12f" % (np.linalg.norm(t1_caea)))
    nevpt.log.info("Correlation energy [0']:                     %20.12f" % e_0p)
    nevpt.log.timer("computing T[0']^(1) amplitudes", *cput0)

    if e_0p > 0.0:
        nevpt.log.info("\n\nWARNING: Positive correlation energy detected in class [0'] !!!!\n\n")
    return e_0p, (t1_caea, t1_caae), K_caca, S_0p_act, S_0p_12_inv_act, V_0p, ic_coeff, (t1_caea_, t1_caae_)

def compute_t1_p1p(nevpt):
    cput0 = (logger.process_clock(), logger.perf_counter())
    nevpt.log.debug("\nComputing T[+1']^(1) amplitudes...")

    ncore = nevpt.ncore
    ncas = nevpt.ncas
    e_core = nevpt.mo_energy.c
    h_ca = nevpt.h1eff.ca
    v_caaa = nevpt.v2e.caaa
    rdms = nevpt.rdms
    rdm_ca = rdms.ca
    rdm_ccaa = rdms.ccaa
    rdm_cccaaa = rdms.cccaaa

    K_p1p = intermediates.compute_K_p1p(nevpt)
    S_p1p_act, S_p1p_12_inv_act = overlap.compute_S12_p1p(nevpt)
    # SKS = reduce(np.dot, (S_p1p_12_inv_act.T, K_p1p, S_p1p_12_inv_act))
    # evals, evecs = np.linalg.eigh(SKS)
    # del(SKS)
    V2_aa_aa  = 1/2 * einsum('IY,XZ->IXYZ', h_ca, rdm_ca)
    V2_aa_aa -= 1/2 * einsum('IZ,XY->IXYZ', h_ca, rdm_ca)
    V2_aa_aa += 1/6 * einsum('Ix,XxYZ->IXYZ', h_ca, rdm_ccaa)
    V2_aa_aa -= 1/6 * einsum('Ix,XxZY->IXYZ', h_ca, rdm_ccaa)
    V2_aa_aa += 1/2 * einsum('IYxZ,Xx->IXYZ', v_caaa, rdm_ca)
    V2_aa_aa += 1/2 * einsum('IYxy,XyZx->IXYZ', v_caaa, rdm_ccaa)
    V2_aa_aa -= 1/2 * einsum('IZxY,Xx->IXYZ', v_caaa, rdm_ca)
    V2_aa_aa -= 1/2 * einsum('IZxy,XyYx->IXYZ', v_caaa, rdm_ccaa)
    V2_aa_aa -= 1/6 * einsum('IxyY,XxZy->IXYZ', v_caaa, rdm_ccaa)
    V2_aa_aa += 1/6 * einsum('IxyY,XxyZ->IXYZ', v_caaa, rdm_ccaa)
    V2_aa_aa += 1/6 * einsum('IxyZ,XxYy->IXYZ', v_caaa, rdm_ccaa)
    V2_aa_aa -= 1/6 * einsum('IxyZ,XxyY->IXYZ', v_caaa, rdm_ccaa)
    V2_aa_aa += 1/6 * einsum('Ixyz,XxzYZy->IXYZ', v_caaa, rdm_cccaaa)
    V2_aa_aa -= 1/6 * einsum('Ixyz,XxzZYy->IXYZ', v_caaa, rdm_cccaaa)
    V2_ab_ba =- 1/2 * einsum('IZ,XY->IXYZ', h_ca, rdm_ca)
    V2_ab_ba += 1/3 * einsum('Ix,XxYZ->IXYZ', h_ca, rdm_ccaa)
    V2_ab_ba += 1/6 * einsum('Ix,XxZY->IXYZ', h_ca, rdm_ccaa)
    V2_ab_ba -= 1/2 * einsum('IZxY,Xx->IXYZ', v_caaa, rdm_ca)
    V2_ab_ba -= 1/2 * einsum('IZxy,XyYx->IXYZ', v_caaa, rdm_ccaa)
    V2_ab_ba += 1/6 * einsum('IxyY,XxZy->IXYZ', v_caaa, rdm_ccaa)
    V2_ab_ba += 1/3 * einsum('IxyY,XxyZ->IXYZ', v_caaa, rdm_ccaa)
    V2_ab_ba += 1/3 * einsum('IxyZ,XxYy->IXYZ', v_caaa, rdm_ccaa)
    V2_ab_ba += 1/6 * einsum('IxyZ,XxyY->IXYZ', v_caaa, rdm_ccaa)
    V2_ab_ba += 1/3 * einsum('Ixyz,XxzYZy->IXYZ', v_caaa, rdm_cccaaa)
    V2_ab_ba += 1/6 * einsum('Ixyz,XxzZYy->IXYZ', v_caaa, rdm_cccaaa)

    tril_ind = np.tril_indices(ncas, k=-1)
    V2_aa_aa = V2_aa_aa[:,:,tril_ind[0],tril_ind[1]]
    V2_aa_aa = V2_aa_aa.reshape(ncore, -1)
    V2_ab_ba = V2_ab_ba.reshape(ncore, -1)

    dim_YWZ = ncas * ncas * ncas
    dim_tril_YWZ = ncas * ncas * (ncas - 1) // 2
    dim_act = dim_tril_YWZ + dim_YWZ
    V_aaa_i = 0
    V_aaa_f = V_aaa_i + dim_tril_YWZ
    V_bba_i = V_aaa_f
    V_bba_f = V_bba_i + dim_YWZ
    V_p1p = np.zeros((ncore, dim_act))
    V_p1p[:,V_aaa_i:V_aaa_f] = V2_aa_aa.copy()
    V_p1p[:,V_bba_i:V_bba_f] = V2_ab_ba.copy()
    del(V2_aa_aa, V2_ab_ba)
    # d_ip = (-e_core[:,None] + evals)
    # d_ip = d_ip**(-1)
    # S_12_V_p1p = einsum("iP,Pm->im", V_p1p, S_p1p_12_inv_act)
    # S_12_V_p1p = einsum("mp,im->ip", evecs, S_12_V_p1p)
    # S_12_V_p1p = einsum("ip,ip->ip", d_ip, S_12_V_p1p)
    # S_12_V_p1p = einsum("mp,ip->im", evecs, S_12_V_p1p)
    # del(d_ip)
    # t_p1p = einsum("Pm,im->iP", S_p1p_12_inv_act, S_12_V_p1p)
    # del(S_12_V_p1p)
    # t1_caaa = t_p1p[:,V_bba_i: V_bba_f].reshape(ncore, ncas, ncas, ncas).copy()
    # t1_caaa = t1_caaa.transpose(0,1,3,2).copy()
    # del(t_p1p)
    # e_p1p = -einsum('ix,iyzw,xyzw', h_ca, t1_caaa, rdm_ccaa)
    # e_p1p -= einsum('ix,iyzx,zy', h_ca, t1_caaa, rdm_ca)
    # e_p1p += 2 * einsum('ix,iyxz,zy', h_ca, t1_caaa, rdm_ca)
    # e_p1p += 1/3 * einsum('ixyz,iwuv,xwvyzu', t1_caaa, v_caaa, rdm_cccaaa)
    # e_p1p -= 2/3 * einsum('ixyz,iwuv,xwvzyu', t1_caaa, v_caaa, rdm_cccaaa)
    # e_p1p -= einsum('ixyz,iwuz,yuwx', t1_caaa, v_caaa, rdm_ccaa)
    # e_p1p -= einsum('ixyz,izwu,ywxu', t1_caaa, v_caaa, rdm_ccaa)
    # e_p1p += 1/3 * einsum('ixyz,iwuv,xwvuzy', t1_caaa, v_caaa, rdm_cccaaa)
    # e_p1p += 1/3 * einsum('ixyz,iwuv,xwvuyz', t1_caaa, v_caaa, rdm_cccaaa)
    # e_p1p += 1/3 * einsum('ixyz,iwuv,xwvzuy', t1_caaa, v_caaa, rdm_cccaaa)
    # e_p1p += 1/3 * einsum('ixyz,iwuv,xwvyuz', t1_caaa, v_caaa, rdm_cccaaa)
    # e_p1p -= einsum('ixyz,iwuy,zuxw', t1_caaa, v_caaa, rdm_ccaa)
    # e_p1p -= einsum('ixyz,izwy,xw', t1_caaa, v_caaa, rdm_ca)
    # e_p1p += 2 * einsum('ixyz,iywu,zwxu', t1_caaa, v_caaa, rdm_ccaa)
    # e_p1p += 2 * einsum('ixyz,iywz,xw', t1_caaa, v_caaa, rdm_ca)

    Delta = -e_core
    ic_coeff = solve_amplitude(K_p1p, S_p1p_12_inv_act, V_p1p, Delta)
    t_p1p = einsum('...m,Pm->...P', ic_coeff, S_p1p_12_inv_act)
    t1_caaa = t_p1p[:,V_bba_i: V_bba_f].reshape(ncore, ncas, ncas, ncas).copy()
    t1_caaa = t1_caaa.transpose(0,1,3,2).copy()
    t1_caaa_ = -einsum('ix,xyzw->iyzw', h_ca, rdm_ccaa)
    t1_caaa_ -= einsum('ix,zy->iyzx', h_ca, rdm_ca)
    t1_caaa_ += 2 * einsum('ix,zy->iyxz', h_ca, rdm_ca)
    t1_caaa_ += 1/3 * einsum('iwuv,xwvyzu->ixyz', v_caaa, rdm_cccaaa)
    t1_caaa_ -= 2/3 * einsum('iwuv,xwvzyu->ixyz', v_caaa, rdm_cccaaa)
    t1_caaa_ -= einsum('iwuz,yuwx->ixyz', v_caaa, rdm_ccaa)
    t1_caaa_ -= einsum('izwu,ywxu->ixyz', v_caaa, rdm_ccaa)
    t1_caaa_ += 1/3 * einsum('iwuv,xwvuzy->ixyz', v_caaa, rdm_cccaaa)
    t1_caaa_ += 1/3 * einsum('iwuv,xwvuyz->ixyz', v_caaa, rdm_cccaaa)
    t1_caaa_ += 1/3 * einsum('iwuv,xwvzuy->ixyz', v_caaa, rdm_cccaaa)
    t1_caaa_ += 1/3 * einsum('iwuv,xwvyuz->ixyz', v_caaa, rdm_cccaaa)
    t1_caaa_ -= einsum('iwuy,zuxw->ixyz', v_caaa, rdm_ccaa)
    t1_caaa_ -= einsum('izwy,xw->ixyz',  v_caaa, rdm_ca)
    t1_caaa_ += 2 * einsum('iywu,zwxu->ixyz', v_caaa, rdm_ccaa)
    t1_caaa_ += 2 * einsum('iywz,xw->ixyz', v_caaa, rdm_ca)
    e_p1p = einsum('ixyz,ixyz', t1_caaa, t1_caaa_)

    nevpt.log.debug("Norm of T[+1']^(1):                          %20.12f" % (np.linalg.norm(t1_caaa)))
    nevpt.log.info("Correlation energy [+1']:                    %20.12f" % e_p1p)
    nevpt.log.timer("computing T[+1']^(1) amplitudes", *cput0)

    if e_p1p > 0.0:
        nevpt.log.info("\n\nWARNING: Positive correlation energy detected in class [+1'] !!!!\n\n")
    return e_p1p, t1_caaa, K_p1p, S_p1p_act, S_p1p_12_inv_act, V_p1p, ic_coeff, t1_caaa_

def compute_t1_m1p(nevpt):
    cput0 = (logger.process_clock(), logger.perf_counter())
    nevpt.log.debug("\nComputing T[-1']^(1) amplitudes...")

    ncas = nevpt.ncas
    nextern = nevpt.nextern
    e_extern = nevpt.mo_energy.e
    h_ae = nevpt.h1eff.ae
    v_aaae = nevpt.v2e.aaae
    rdms = nevpt.rdms
    rdm_ca = rdms.ca
    rdm_ccaa = rdms.ccaa
    rdm_cccaaa = rdms.cccaaa

    K_m1p = intermediates.compute_K_m1p(nevpt)
    S_m1p_act, S_m1p_12_inv_act = overlap.compute_S12_m1p(nevpt)
    # SKS = reduce(np.dot, (S_m1p_12_inv_act.T, K_m1p, S_m1p_12_inv_act))
    # evals, evecs = np.linalg.eigh(SKS)
    # del(SKS)
    V2_aa_aa  = 1/6 * einsum('xA,XYZx->XYZA', h_ae, rdm_ccaa)
    V2_aa_aa -= 1/6 * einsum('xA,XYxZ->XYZA', h_ae, rdm_ccaa)
    V2_aa_aa += 1/6 * einsum('xZyA,XYxy->XYZA', v_aaae, rdm_ccaa)
    V2_aa_aa -= 1/6 * einsum('xZyA,XYyx->XYZA', v_aaae, rdm_ccaa)
    V2_aa_aa += 1/6 * einsum('xyzA,XYyZzx->XYZA', v_aaae, rdm_cccaaa)
    V2_aa_aa -= 1/6 * einsum('xyzA,XYyzZx->XYZA', v_aaae, rdm_cccaaa)
    V2_ab_ba =- 1/6 * einsum('xA,XYZx->XYZA', h_ae, rdm_ccaa)
    V2_ab_ba -= 1/3 * einsum('xA,XYxZ->XYZA', h_ae, rdm_ccaa)
    V2_ab_ba -= 1/6 * einsum('xZyA,XYxy->XYZA', v_aaae, rdm_ccaa)
    V2_ab_ba -= 1/3 * einsum('xZyA,XYyx->XYZA', v_aaae, rdm_ccaa)
    V2_ab_ba += 1/6 * einsum('xyzA,XYyZxz->XYZA', v_aaae, rdm_cccaaa)
    V2_ab_ba += 1/6 * einsum('xyzA,XYyxZz->XYZA', v_aaae, rdm_cccaaa)
    V2_ab_ba += 1/6 * einsum('xyzA,XYyxzZ->XYZA', v_aaae, rdm_cccaaa)
    V2_ab_ba -= 1/6 * einsum('xyzA,XYyzZx->XYZA', v_aaae, rdm_cccaaa)
    V2_ab_ba += 1/6 * einsum('xyzA,XYyzxZ->XYZA', v_aaae, rdm_cccaaa)

    tril_ind = np.tril_indices(ncas, k=-1)
    V2_aa_aa = V2_aa_aa[tril_ind[0], tril_ind[1]]
    V2_aa_aa = V2_aa_aa.reshape(-1, nextern)
    V2_ab_ba = V2_ab_ba.reshape(-1, nextern)

    dim_YWZ = ncas * ncas * ncas
    dim_tril_YWZ = ncas * ncas * (ncas - 1) // 2
    dim_act = dim_tril_YWZ + dim_YWZ
    V_aaa_i = 0
    V_aaa_f = V_aaa_i + dim_tril_YWZ
    V_abb_i = V_aaa_f
    V_abb_f = V_abb_i + dim_YWZ
    V_m1p = np.zeros((dim_act, nextern))
    V_m1p[V_aaa_i:V_aaa_f] = V2_aa_aa.copy()
    V_m1p[V_abb_i:V_abb_f] = V2_ab_ba.copy()
    V_m1p = V_m1p.T
    del(V2_aa_aa, V2_ab_ba)
    # d_pa = (evals[:,None] + e_extern)
    # d_pa = d_pa**(-1)
    # S_12_V_m1p = einsum("Pa,Pm->ma", V_m1p, S_m1p_12_inv_act)
    # S_12_V_m1p = einsum("mp,ma->pa", evecs, S_12_V_m1p)
    # S_12_V_m1p = einsum("pa,pa->pa", d_pa, S_12_V_m1p)
    # S_12_V_m1p = einsum("mp,pa->ma", evecs, S_12_V_m1p)
    # del(d_pa)
    # t_m1p = einsum("Pm,ma->Pa", S_m1p_12_inv_act, S_12_V_m1p)
    # del(S_12_V_m1p)
    # t1_aaae = t_m1p[V_abb_i:V_abb_f, :].reshape(ncas, ncas, ncas, nextern).copy()
    # t1_aaae = t1_aaae.transpose(1,0,2,3)
    # del(t_m1p)
    # e_m1p = einsum('xa,yzwa,xwzy', h_ae, t1_aaae, rdm_ccaa)
    # e_m1p -= 1/3 * einsum('xyza,wuva,zvwuyx', t1_aaae, v_aaae, rdm_cccaaa)
    # e_m1p -= 1/3 * einsum('xyza,wuva,zvwxuy', t1_aaae, v_aaae, rdm_cccaaa)
    # e_m1p += 2/3 * einsum('xyza,wuva,zvwxyu', t1_aaae, v_aaae, rdm_cccaaa)
    # e_m1p -= 1/3 * einsum('xyza,wuva,zvwuxy', t1_aaae, v_aaae, rdm_cccaaa)
    # e_m1p -= 1/3 * einsum('xyza,wuva,zvwyux', t1_aaae, v_aaae, rdm_cccaaa)
    # e_m1p -= 1/3 * einsum('xyza,wuva,zvwyxu', t1_aaae, v_aaae, rdm_cccaaa)
    # e_m1p += einsum('xyza,wzua,xywu', t1_aaae, v_aaae, rdm_ccaa)

    Delta = e_extern
    ic_coeff = solve_amplitude(K_m1p, S_m1p_12_inv_act, V_m1p, Delta)
    t_m1p = einsum('...m,Pm->...P', ic_coeff, S_m1p_12_inv_act).T
    t1_aaae = t_m1p[V_abb_i:V_abb_f, :].reshape(ncas, ncas, ncas, nextern).copy()
    t1_aaae = t1_aaae.transpose(1,0,2,3)
    t1_aaae_ = einsum('xa,xwzy->yzwa', h_ae, rdm_ccaa)
    t1_aaae_ -= 1/3 * einsum('wuva,zvwuyx->xyza', v_aaae, rdm_cccaaa)
    t1_aaae_ -= 1/3 * einsum('wuva,zvwxuy->xyza', v_aaae, rdm_cccaaa)
    t1_aaae_ += 2/3 * einsum('wuva,zvwxyu->xyza', v_aaae, rdm_cccaaa)
    t1_aaae_ -= 1/3 * einsum('wuva,zvwuxy->xyza', v_aaae, rdm_cccaaa)
    t1_aaae_ -= 1/3 * einsum('wuva,zvwyux->xyza', v_aaae, rdm_cccaaa)
    t1_aaae_ -= 1/3 * einsum('wuva,zvwyxu->xyza', v_aaae, rdm_cccaaa)
    t1_aaae_ += einsum('wzua,xywu->xyza', v_aaae, rdm_ccaa)
    e_m1p = einsum('xyza,xyza', t1_aaae, t1_aaae_)

    nevpt.log.debug("Norm of T[-1']^(1):                          %20.12f" % (np.linalg.norm(t1_aaae)))
    nevpt.log.info("Correlation energy [-1']:                    %20.12f" % e_m1p)
    nevpt.log.timer("computing T[-1']^(1) amplitudes", *cput0)

    if e_m1p > 0.0:
        nevpt.log.info("\n\nWARNING: Positive correlation energy detected in class [-1'] !!!!\n\n")
    return e_m1p, t1_aaae, K_m1p, S_m1p_act, S_m1p_12_inv_act, V_m1p, ic_coeff, t1_aaae_