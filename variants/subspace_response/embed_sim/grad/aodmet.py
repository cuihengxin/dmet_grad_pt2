from pyscf import mcscf, fci, ao2mo, lib
from pyscf.lib import logger
from functools import reduce
import numpy as np
import scipy
from embed_sim import aodmet
from embed_sim.grad.ssdmet import (get_es_imds, gen_get_indep_pair_slice, get_S_1, get_H_1, group_indexes_by_diff, get_grad_dm1, get_grad_veff, get_grad_dm2, get_I, get_grad_X, gen_contract_Z_A, get_grad_Z_vector)

from functools import partial
from numpy import einsum
einsum = partial(einsum, optimize=True)

def mol_slice(atm_id, mol):
    _, _, p0, p1 = mol.aoslice_by_atom()[atm_id]
    return slice(p0, p1)

def get_orth_1(mf, orth, imp_idx, S_1_ao, log):
    if orth.lower() == 'lowdin':
        return get_lowdin_1(mf, imp_idx, S_1_ao, log)
    elif orth.lower() == 'cholesky':
        return get_cholesky_1(mf, imp_idx, S_1_ao, log)
    else:
        raise NotImplementedError

# def get_lowdin_1(mf, imp_idx, S_1_ao, log):
#     t0 = (logger.process_clock(), logger.perf_counter())
    
#     nao = mf.mol.nao
#     natm = mf.mol.natm
#     env_idx = np.array([x for x in range(nao) if x not in imp_idx])
#     rearange_idx = np.argsort(np.concatenate((imp_idx, env_idx)))
#     nenv = len(env_idx)
#     nimp = len(imp_idx)
#     S = mf.get_ovlp()
#     S_env = S[env_idx,:][:,env_idx]
#     sval,svec = np.linalg.eigh(S_env)
#     Ws = sval[:,None] - sval[None,:]
#     Ws[range(nenv), range(nenv)] = np.inf
#     if (abs(Ws) < 1e-12).sum() > 0:
#         if (abs(Ws) < 1e-15).sum() > 0:
#             log.warn(f'''
#     {(abs(Ws) < 1e-15).sum()} degenerate eigenvalues of the environmental overlap matrix are found,
#     the gradients are unreliable! Please fine-tune your initial structure!
#     ''')
#         else:
#             log.warn(f'''
#     {(abs(Ws) < 1e-12).sum()} nearly degenerate eigenvalues of the environmental overlap matrix are found,
#     the gradients may suffer from numerical instability.
#     ''')
#     Ws[abs(Ws) < 1e-15] = np.inf
#     Ws = 1 / Ws
#     S_1_ao_env = S_1_ao[:,:,env_idx,:][:,:,:,env_idx]
#     S_1_svec = einsum('Atab,aj,bm->Atjm',S_1_ao_env,svec,svec)
#     nondegen_sval, degen_sval_list = group_indexes_by_diff(sval, threshold=1e-15)
#     svec_nondegen = svec[:,nondegen_sval]
#     Ws_nondegen = Ws[nondegen_sval]
#     sval_nondegen = sval[nondegen_sval]
#     S_1_svec_nondegen = S_1_svec[:,:,:,nondegen_sval]
#     caolo_1_env = (
#         einsum('mj,pj,rm,Atjm->Atpr',Ws_nondegen,svec,svec_nondegen*(sval_nondegen**(-1/2)),S_1_svec_nondegen)
#     )
#     caolo_1_env += caolo_1_env.swapaxes(-1,-2)
#     caolo_1_env -= 0.5*einsum('pm,rm,Atmm->Atpr',
#                               svec_nondegen,svec_nondegen*(sval_nondegen**(-3/2)),S_1_svec_nondegen[:,:,nondegen_sval])
#     cloao_1_env = (
#         einsum('mj,pj,rm,Atjm->Atpr',Ws_nondegen,svec,svec_nondegen*(sval_nondegen**(1/2)),S_1_svec_nondegen)
#     )
#     cloao_1_env += cloao_1_env.swapaxes(-1,-2)
#     cloao_1_env += 0.5*einsum('pm,rm,Atmm->Atpr',
#                               svec_nondegen,svec_nondegen*(sval_nondegen**(-1/2)),S_1_svec_nondegen[:,:,nondegen_sval])
#     for degen_sval in degen_sval_list:
#         degen_sval_comp = [x for x in range(nenv) if x not in degen_sval]
#         svec_degen = svec[:,degen_sval]
#         svec_degen_comp = svec[:,degen_sval_comp]
#         Ws_degen = Ws[degen_sval,:][:,degen_sval_comp]
#         sval_degen = sval[degen_sval]
#         S_1_svec_degen_comp = einsum('Atab,aj,bm->Atjm',S_1_ao_env,svec_degen_comp,svec_degen)
#         S_1_svec_degen = einsum('Atab,aj,bm->Atjm',S_1_ao_env,svec_degen,svec_degen)
#         caolo_1_degen = einsum('mj,pj,rm,Atjm->Atpr',
#                                Ws_degen,svec_degen_comp,svec_degen*(sval_degen**(-1/2)),S_1_svec_degen_comp)
#         caolo_1_degen += caolo_1_degen.swapaxes(-1,-2)
#         caolo_1_degen -= 0.5*einsum('pm,rm,Atmm->Atpr',
#                                     svec_degen,svec_degen*(sval_degen**(-3/2)),S_1_svec_degen)
#         cloao_1_degen = einsum('mj,pj,rm,Atjm->Atpr',
#                                Ws_degen,svec_degen_comp,svec_degen*(sval_degen**(1/2)),S_1_svec_degen_comp)
#         cloao_1_degen += cloao_1_degen.swapaxes(-1,-2)
#         cloao_1_degen += 0.5*einsum('pm,rm,Atmm->Atpr',
#                                     svec_degen,svec_degen*(sval_degen**(-1/2)),S_1_svec_degen)
#         caolo_1_env += caolo_1_degen
#         cloao_1_env += cloao_1_degen
#     caolo_1 = np.zeros((natm, 3, nao, nao))
#     cloao_1 = np.zeros((natm, 3, nao, nao))
#     caolo_1[:,:,nimp:,nimp:] += caolo_1_env
#     cloao_1[:,:,nimp:,nimp:] += cloao_1_env
#     cloao_1 = cloao_1[:,:,rearange_idx,:][:,:,:,rearange_idx]
#     caolo_1 = caolo_1[:,:,rearange_idx,:][:,:,:,rearange_idx]
    
#     t0 = log.timer('lowdin_1', *t0)
#     return caolo_1, cloao_1

def get_lowdin_1(mf, imp_idx, S_1_ao, log):
    from pyscf.lo.orth import lowdin
    t0 = (logger.process_clock(), logger.perf_counter())
    
    nao = mf.mol.nao
    natm = mf.mol.natm
    env_idx = np.array([x for x in range(nao) if x not in imp_idx])
    rearange_idx = np.argsort(np.concatenate((imp_idx, env_idx)))
    nenv = len(env_idx)
    nimp = len(imp_idx)
    S = mf.get_ovlp()
    S_env = S[env_idx,:][:,env_idx]
    S_1_ao_env = S_1_ao[:,:,env_idx,:][:,:,:,env_idx]

    caolo_env = lowdin(S_env)
    cloao_env = lib.dot(lowdin(S_env), S_env)
    Z, Y = np.linalg.eigh(cloao_env)
    YBY = einsum('iu,Atij,jv,uv->Atuv', Y, S_1_ao_env, Y, 1/lib.direct_sum('u+v->uv', Z, Z))
    cloao_1_env = einsum('iu,Atuv,jv->Atij', Y, YBY, Y)
    caolo_1_env = -1 * einsum('ui,Atij,vj->Atuv', caolo_env, cloao_1_env, caolo_env)

    caolo_1 = np.zeros((natm, 3, nao, nao))
    cloao_1 = np.zeros((natm, 3, nao, nao))
    caolo_1[:,:,nimp:,nimp:] += caolo_1_env
    cloao_1[:,:,nimp:,nimp:] += cloao_1_env
    cloao_1 = cloao_1[:,:,rearange_idx,:][:,:,:,rearange_idx]
    caolo_1 = caolo_1[:,:,rearange_idx,:][:,:,:,rearange_idx]
    
    t0 = log.timer('lowdin_1', *t0)
    return caolo_1, cloao_1

def get_cholesky_1(mf, imp_idx, S_1_ao, log):
    from embed_sim.grad.chol_diff import chol_fwd

    t0 = (logger.process_clock(), logger.perf_counter())

    mol = mf.mol
    nao = mol.nao
    natm = mol.natm
    env_idx = np.array([x for x in range(nao) if x not in imp_idx])
    rearange_idx = np.argsort(np.concatenate((imp_idx, env_idx)))
    nenv = len(env_idx)
    nimp = len(imp_idx)
    S = mf.get_ovlp()
    S_env = S[env_idx,:][:,env_idx]
    L = np.linalg.cholesky(S_env)
    caolo = np.linalg.inv(L.T)
    cloao = L.T
    
    S_1_tril = S_1_ao[:,:,env_idx,:][:,:,:,env_idx].copy()
    S_1_tril[:,:,range(nenv),range(nenv)] *= 0.5
    cloao_1_env = np.array([[chol_fwd(L, np.tril(S_1_tril[i,j])) for j in range(3)] for i in range(natm)])

    caolo_1 = np.zeros((natm, 3, nao, nao))
    cloao_1 = np.zeros((natm, 3, nao, nao))
    
    cloao_1[:,:,nimp:,nimp:] += cloao_1_env.swapaxes(-1,-2)
    caolo_1[:,:,nimp:,nimp:] -= einsum('mn,Atpn,pq->Atmq', caolo, cloao_1_env, caolo)
    cloao_1 = cloao_1[:,:,rearange_idx,:][:,:,:,rearange_idx]
    caolo_1 = caolo_1[:,:,rearange_idx,:][:,:,:,rearange_idx]
    
    t0 = log.timer('cholesky_1', *t0)
    return caolo_1, cloao_1

def get_grad_I(mydmet_grad, I1, I2, caolo_1, cloao_1, S_1_mo, log):
    t0 = (logger.process_clock(), logger.perf_counter())
    
    mydmet = mydmet_grad.base
    mf = mydmet.mf_or_cas
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    mol = mf.mol
    nao = mol.nao
    ndo = (mo_occ==2).sum()
    nho = ((mo_occ>0) & (mo_occ<2)).sum()
    nvo = (mo_occ==0).sum()
    nocc = ndo + nho
    nmo = nao
    nvir = nmo - nocc
    nvo = nvir
    n_indep_pair = nho*ndo + nvo*ndo + nvo*nho
    sd, sh, so, sv, sa = slice(0, ndo), slice(ndo, nocc), slice(0, nocc), slice(nocc, nmo), slice(0, nmo)
    mo_coeff_occ = mo_coeff[:, mo_occ>0]
    
    ldm, caolo, cloao = mydmet.lowdin_orth()
    imp_idx = mydmet.imp_idx
    env_idx = np.array([x for x in range(ldm.shape[0]) if x not in imp_idx])
    rearange_idx = np.argsort(np.concatenate((imp_idx, env_idx)))
    ldm_env = ldm[env_idx,:][:,env_idx]
    occ_env, orb_env = np.linalg.eigh(ldm_env)
    thres = mydmet.threshold
    thres2 = mydmet.threshold2
    nimp = len(imp_idx)
    vir_idx = (occ_env < thres)
    bath_idx = (((occ_env >= thres) & (occ_env <= 2-thres2)) | (occ_env >= 2+thres2))
    core_idx = ((occ_env > 2-thres2) & (occ_env < 2+thres2))
    nvir = vir_idx.sum()
    nbath = bath_idx.sum()
    ncore = core_idx.sum()
    occ_env = np.hstack((occ_env[vir_idx], occ_env[bath_idx], occ_env[core_idx]))
    orb_env = np.hstack((orb_env[:, vir_idx], orb_env[:, bath_idx], orb_env[:, core_idx]))
    lo2eo = scipy.linalg.block_diag(np.eye(nimp),orb_env[:,nvir:nvir+nbath])
    lo2core = np.vstack((np.zeros((nimp,ncore)),orb_env[:,nvir+nbath:]))
    lo2eo = lo2eo[rearange_idx,:]
    lo2core = lo2core[rearange_idx,:]
    
    mf = mydmet.mf_or_cas
    # dma, dmb = mf.make_rdm1()
    # dm = dma + dmb
    dm = mydmet.dm
    cloao_env_dm = einsum('nl,sl->ns', cloao[env_idx,:][:,env_idx], dm[env_idx,:][:,env_idx])
    core_vec_comp = orb_env[:,:nvir+nbath]
    core_vec = orb_env[:,nvir+nbath:]
    bath_vec = orb_env[:,nvir:nvir+nbath]
    from embed_sim.grad._embedding_response import response_factors
    W_bath, W_core = response_factors(
        mydmet_grad, occ_env, nvir, nbath, log, implementation="aodmet")
    grad_I1 = (
        + einsum('uj,uv,k,mk,nj,vk,ns,Atms->At',
                 I1,caolo[:,env_idx],W_core,core_vec_comp,core_vec,core_vec_comp,cloao_env_dm,cloao_1[:,:,env_idx,:][:,:,:,env_idx])
        + einsum('uj,uv,k,nk,mj,vk,ns,Atms->At',
                 I1,caolo[:,env_idx],W_core,core_vec_comp,core_vec,core_vec_comp,cloao_env_dm,cloao_1[:,:,env_idx,:][:,:,:,env_idx])
        + einsum('uj,Atuv,vj->At',I1,caolo_1,lo2core)
    )
    grad_I2 = (
        + einsum('uj,uv,jk,mk,nj,vk,ns,Atms->At',
                 I2[:,nimp:],caolo[:,env_idx],W_bath,orb_env,bath_vec,orb_env,cloao_env_dm,cloao_1[:,:,env_idx,:][:,:,:,env_idx])
        + einsum('uj,uv,jk,nk,mj,vk,ns,Atms->At',
                 I2[:,nimp:],caolo[:,env_idx],W_bath,orb_env,bath_vec,orb_env,cloao_env_dm,cloao_1[:,:,env_idx,:][:,:,:,env_idx])
        + einsum('up,Atuv,vp->At',I2,caolo_1,lo2eo)
    )
    grad_I = grad_I1 + grad_I2
    
    caolo_env = caolo[:,env_idx]
    cloao_env = cloao[env_idx,:]
    Y = (
        + einsum('uj,k,mk,nj,vk,uv,ms,nl,si,lq->qi',
                  I1,W_core,core_vec_comp,core_vec,core_vec_comp,caolo_env,cloao_env,cloao_env,mo_coeff_occ,mo_coeff)
        + einsum('uj,k,nk,mj,vk,uv,ms,nl,si,lq->qi',
                  I1,W_core,core_vec_comp,core_vec,core_vec_comp,caolo_env,cloao_env,cloao_env,mo_coeff_occ,mo_coeff)
        + einsum('uj,jk,mk,nj,vk,uv,ms,nl,si,lq->qi',
                  I2[:,nimp:],W_bath,orb_env,bath_vec,orb_env,caolo_env,cloao_env,cloao_env,mo_coeff_occ,mo_coeff)
        + einsum('uj,jk,nk,mj,vk,uv,ms,nl,si,lq->qi',
                  I2[:,nimp:],W_bath,orb_env,bath_vec,orb_env,caolo_env,cloao_env,cloao_env,mo_coeff_occ,mo_coeff)
    )
    if hasattr(mf, 'frac'):
        f = 2 * mf.frac
    else:
        f = 1
    Y_tilde_hd = (2*Y[sh,sd] - f*Y[sd,sh].T).flatten()
    Y_tilde_vd = 2*Y[sv,sd].flatten()
    Y_tilde_vh = f*Y[sv,sh].flatten()
    W = np.concatenate((Y_tilde_hd,Y_tilde_vd,Y_tilde_vh))
    grad_I += (
        - einsum('ij,Atij->At',Y[sd,sd],S_1_mo[:,:,sd,sd])
        - 0.5*f*einsum('uv,Atuv->At',Y[sh,sh],S_1_mo[:,:,sh,sh])
        - f*einsum('iu,Atui->At',Y[sd,sh],S_1_mo[:,:,sh,sd])
    )
    
    t0 = log.timer('grad_I', *t0)
    return grad_I, W

from pyscf.lib import logger

def grad_elec(mydmet_grad, atmlst=None):
    mydmet = mydmet_grad.base
    solver_option = mydmet_grad.solver_option
    from embed_sim.grad.guess import prepare_options
    solver_option = prepare_options(mydmet, solver_option)
    mf = mydmet.mf_or_cas
    mo_coeff = mf.mo_coeff
    mol = mydmet.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    de = np.zeros((len(atmlst),3))
    log = logger.Logger(mydmet_grad.stdout, mydmet_grad.verbose)
    
    es_mf = mydmet.es_mf
    fo_ene = mydmet.fo_ene
    ao2eo = mydmet.es_orb
    ao2core = mydmet.fo_orb
    dm_core = lib.dot(ao2core, ao2core.T)*2
    neo = ao2eo.shape[-1]
    nao = mol.nao
    get_indep_pair_slice = gen_get_indep_pair_slice(mf)
    
    es_solver, es_ene, es_relaxed_rdm1, es_relaxed_rdm2, es_relaxed_X = get_es_imds(es_mf, solver_option, log)
    S_1_ao, S_1_mo = get_S_1(mf, atmlst, log)
    H_1_ao, H_1_mo = get_H_1(mf, atmlst, log)
    caolo_1, cloao_1 = get_orth_1(mf, mydmet.orth, mydmet.imp_idx, S_1_ao, log)
    epsilon_pq = mf.mo_energy[:,None] - mf.mo_energy[None,:]
    epsilon_indep_pair = get_indep_pair_slice(epsilon_pq)
    es_relaxed_rdm1_global = reduce(lib.dot, (ao2eo, es_relaxed_rdm1, ao2eo.T))
    I1, I2 = get_I(mf, dm_core, ao2eo, ao2core, es_relaxed_rdm1_global, es_relaxed_rdm1, es_relaxed_rdm2, es_relaxed_X, log)
    grad_I, W = get_grad_I(mydmet_grad, I1, I2, caolo_1, cloao_1, S_1_mo, log)
    
    t1 = (logger.process_clock(), logger.perf_counter())
    contract_Z_A = gen_contract_Z_A(mf, cache_static=mydmet_grad.zvector_cache_static)
    from embed_sim.grad._zvector import solve_zvector
    Z = solve_zvector(mydmet_grad, mf, W, epsilon_indep_pair,
                      contract_Z_A, get_indep_pair_slice, log)
    t1 = log.timer('Z-vector equation', *t1)
    
    de += get_grad_dm1(H_1_ao, dm_core, es_relaxed_rdm1_global, log)
    de += get_grad_veff(mf, atmlst, dm_core, es_relaxed_rdm1_global, log)
    de += get_grad_dm2(mf, atmlst, ao2eo, es_relaxed_rdm2, log)
    de += get_grad_X(S_1_ao, ao2eo, es_relaxed_X, log)
    de += grad_I
    de += get_grad_Z_vector(mf, atmlst, Z, H_1_mo, S_1_mo, contract_Z_A, get_indep_pair_slice, log)
    e = es_ene + fo_ene

    del es_relaxed_rdm2
    return de, e, es_solver

from pyscf.grad.rhf import GradientsBase
class AODMET_Gradients(GradientsBase):
    def __init__(self, method):
        self.verbose = method.verbose
        self.stdout = method.stdout
        self.mol = method.mol
        self.base = method
        self.max_memory = self.base.max_mem
        self.unit = 'au'
        self.solver_option = None
        self.es_ene = None
        self._krylov_conv_tol = 1e-7
        self.zvector_solver = "gmres"  # "krylov" retains the original algorithm
        self.zvector_guess_mode = "previous"  # "history": minimum-residual combination
        self.zvector_subspace_tol = None  # default: _krylov_conv_tol
        self.zvector_recycle = False
        self.zvector_recycle_dim = 8
        self.zvector_recycle_min_iterations = 12
        self.zvector_cache_static = True
        self.zvector_reuse_residual = True
        self.zvector_warm_start = True
        self.zvector_restart = 80
        self.zvector_max_cycle = 200  # total inner iterations, not restart cycles
        self.embedding_response = "subspace"  # "legacy" reproduces original denominators
        self._perturb_anyway = False
        self._perturbation = 1e-10

        self.atmlst = None
        self.de = None
        
    grad_elec = grad_elec
    
    def kernel(self, atmlst=None):
        cput0 = (logger.process_clock(), logger.perf_counter())

        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        de, e, es_solver = self.grad_elec(atmlst)
        self.de = de + self.grad_nuc(atmlst=atmlst)
        self.e = e
        self.es_solver = es_solver
        cache = self.solver_option.get('guess_cache')
        if cache is not None:
            cache.update(self.base, es_solver, self.solver_option)
        logger.timer(self, 'DMET gradients', *cput0)
        self._finalize()
        return self.de, self.e

AODMET_Grad = AODMET_Gradients

aodmet.AODMET.Gradients = lib.class_as_method(AODMET_Gradients)
