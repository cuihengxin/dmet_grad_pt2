"""Spectral response of the retained bath subspace, independent of its gauge."""
import numpy as np


def embedding_response_factors(occupations, nvir, nbath, perturbation=None):
    """Return factors in [frozen virtual | bath | frozen occupied] order.

    Only bath/external rotations contribute to the retained projector.
    A vanishing external gap has no unique response and raises, rather than
    silently returning the zero pseudoinverse in the proposed implementation.
    Core factors retain the original assumption of occupation exactly two.
    """
    occupations = np.asarray(occupations, dtype=float)
    if occupations.ndim != 1 or not np.all(np.isfinite(occupations)):
        raise ValueError('occupations must be a finite one-dimensional array')
    if not (0 <= nvir <= nvir + nbath <= occupations.size):
        raise ValueError('invalid virtual/bath dimensions')
    if perturbation is not None and (not np.isfinite(perturbation) or perturbation <= 0):
        raise ValueError('perturbation must be finite and positive')
    bath = slice(nvir, nvir + nbath)
    gap = occupations[bath, None] - occupations[None, :]
    external = np.ones(gap.shape, dtype=bool)
    external[:, bath] = False
    core_gap = 2 - occupations[:nvir + nbath]

    def inverse(values, mask):
        result = np.zeros_like(values, dtype=float)
        if perturbation is None:
            if np.any(mask & (values == 0)):
                raise ValueError('Zero external/core gap: embedding subspace response is not unique; '
                                 'change the partition or explicitly enable damping')
            np.divide(1., values, out=result, where=mask)
        else:
            np.divide(values, values**2 + perturbation, out=result, where=mask)
        return result

    core_mask = np.full(core_gap.shape, occupations.size > nvir + nbath, dtype=bool)
    return inverse(gap, external), inverse(core_gap, core_mask)


def response_factors(grad, occupations, nvir, nbath, log, implementation):
    mode = getattr(grad, 'embedding_response', 'subspace')
    if mode == 'legacy':
        from . import _embedding_response_legacy as legacy
        return getattr(legacy, implementation + '_response_factors')(
            grad, occupations, nvir, nbath, log)
    if mode != 'subspace':
        raise ValueError('embedding_response must be "subspace" or "legacy"')
    bath = slice(nvir, nvir + nbath)
    gap = occupations[bath, None] - occupations[None, :]
    external = np.ones(gap.shape, dtype=bool)
    external[:, bath] = False
    if np.any(np.abs(gap[external]) < 1e-8):
        log.warn('Small bath/external gap: retained embedding subspace may be unstable.')
    # No frozen core means no physical core response; avoid spurious 2-n = 0.
    has_core = len(occupations) > nvir + nbath
    if has_core and np.any(np.abs(2 - occupations[:nvir + nbath]) < 1e-8):
        log.warn('Small core/complement gap: frozen-core response may be unstable.')
    damping = grad._perturbation if grad._perturb_anyway else None
    if damping is not None:
        log.warn('Embedding response uses explicit damping %g; gradients are approximate.', damping)
    return embedding_response_factors(occupations, nvir, nbath, damping)
