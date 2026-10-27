"""
grad = mydmet. Gradients()
grad.embedding_response = 'subspace' #or actual_core or legacy
"""
import numpy as np

RESPONSE_METHODS = ('legacy', 'subspace', 'actual_core')


def response_method(value):
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        value = {1: 'legacy', 2: 'subspace', 3: 'actual_core'}.get(int(value))
    if value not in RESPONSE_METHODS:
        raise ValueError('embedding_response must be legacy/1, subspace/2, or actual_core/3')
    return value


def _legacy(grad, occ, nv, nb, log):
    wb = occ[nv:nv+nb, None] - occ[None, :]
    wb[np.arange(nb), np.arange(nv, nv+nb)] = 1
    # Same adjacent-group degeneracy criterion as group_indexes_by_diff.
    groups = np.split(np.arange(nb), np.flatnonzero(abs(np.diff(occ[nv:nv+nb])) > 1e-8)+1) if nb else []
    ndegen = sum(len(group) for group in groups if len(group) > 1)
    core_gap = 2-occ[:nv+nb]
    if grad._perturb_anyway or ndegen:
        if ndegen and not grad._perturb_anyway:
            log.debug1(f'{ndegen} near degenerate bath orbitals are found; '
                       f'{grad._perturbation} perturbation is added to the response denominators.')
        eps = grad._perturbation
        wb = wb/(wb**2+eps)
        wc = core_gap/(core_gap**2+eps)
    else:
        wb = 1/wb
        wc = 1/core_gap
    wb[np.arange(nb), np.arange(nv, nv+nb)] = 0
    return wb, wc


def get_embedding_response(grad, occ_env, nvir, nbath, log):
    """Return bath factors and vector (methods 1/2) or matrix (method 3) core factors.

Method 2 reproduces get_embedding_response in the supplied server ssdmet.py.
No method changes the embedding partition or the energy definition.
"""
    method = response_method(getattr(grad, 'embedding_response', 'legacy'))
    occ = np.asarray(occ_env, dtype=float)
    if occ.ndim != 1 or not np.all(np.isfinite(occ)):
        raise ValueError('Environment eigenvalues must be a finite one-dimensional array')
    if not 0 <= nvir <= nvir+nbath <= occ.size:
        raise ValueError('Invalid environment virtual/bath dimensions')
    if method == 'legacy':
        return _legacy(grad, occ, nvir, nbath, log)

    end = nvir+nbath
    bath = slice(nvir, end)
    gap = occ[bath, None]-occ[None, :]
    external = np.ones(gap.shape, dtype=bool)
    external[:, bath] = False
    has_core = len(occ) > end
    core_gap = (2-occ[:end] if method == 'subspace'
                else occ[None, end:]-occ[:end, None])
    if np.any(abs(gap[external]) < 1e-8):
        log.warn('Small bath/external eigenvalue gap; embedding subspace response may be unstable.')
    if has_core and np.any(abs(core_gap) < 1e-8):
        log.warn('Small core/complement eigenvalue gap; frozen-core response may be unstable.')
    wb = np.zeros_like(gap)
    wc = np.zeros_like(core_gap)
    if grad._perturb_anyway:
        eps = grad._perturbation
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError('Response perturbation must be finite and positive')
        np.divide(gap, gap**2+eps, out=wb, where=external)
        if has_core:
            wc = core_gap/(core_gap**2+eps)
        log.warn('Explicit damping is applied to embedding response; gradients are approximate.')
    else:
        if np.any(gap[external] == 0) or (has_core and np.any(core_gap == 0)):
            raise ValueError('Zero external/core gap: embedding subspace response is not unique')
        np.divide(1., gap, out=wb, where=external)
        if has_core:
            wc = 1/core_gap
    return wb, wc
