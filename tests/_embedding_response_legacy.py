"""Original response denominators, retained verbatim for regression comparison."""
import numpy as np
from embed_sim.grad.ssdmet import group_indexes_by_diff


def legacy_response(mydmet_grad, occ_env, nvir, nbath, log):
    W_bath = occ_env[nvir:nvir+nbath][:,None] - occ_env[None,:]
    W_bath[range(nbath),range(nvir,nvir+nbath)] = 1
    nbath_degenerate = 0
    bath_degenerate_groups = group_indexes_by_diff(occ_env[nvir:nvir+nbath], threshold=1e-8)[1]
    for group in bath_degenerate_groups:
        nbath_degenerate += len(group)
    if mydmet_grad._perturb_anyway:
        W_bath = W_bath/(W_bath**2 + mydmet_grad._perturbation)
        W_bath[range(nbath),range(nvir,nvir+nbath)] = 0
        W_core = (2-occ_env[:nvir+nbath])
        W_core = W_core/(W_core**2 + mydmet_grad._perturbation)
    else:
        if nbath_degenerate > 0:
            log.debug1(f'''
    {nbath_degenerate} near degenerate bath orbitals are found, the gradients may be unreliable.
    {mydmet_grad._perturbation} perturbation is added to the eigenvalues.
    ''')
            W_bath = W_bath/(W_bath**2 + mydmet_grad._perturbation)
            W_bath[range(nbath),range(nvir,nvir+nbath)] = 0
            W_core = (2-occ_env[:nvir+nbath])
            W_core = W_core/(W_core**2 + mydmet_grad._perturbation)
        else:
            W_bath = 1/W_bath
            W_bath[range(nbath),range(nvir,nvir+nbath)] = 0
            W_core = 1/(2-occ_env[:nvir+nbath])
    return W_bath, W_core
