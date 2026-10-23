"""Small singular response directions from a completed Krylov search space."""
import numpy as np


def slow_directions(vectors, images, count):
    """Select small ||A u|| directions in span(vectors), without new A actions.

    These are right singular directions of the restricted operator, not exact
    eigenvectors or harmonic Ritz vectors of the full response matrix.
    """
    if not vectors or count < 1:
        return None
    v = np.column_stack(vectors)
    av = np.column_stack(images)
    norms = np.linalg.norm(v, axis=0)
    keep = np.isfinite(norms) & (norms > 1e-100)
    if not np.any(keep):
        return None
    v, av = v[:, keep]/norms[keep], av[:, keep]/norms[keep]
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(av)):
        return None
    _, s, vt = np.linalg.svd(v, full_matrices=False)
    keep = s > 1e-10*s[0]
    transform = vt[keep].T/s[keep]
    q, aq = v@transform, av@transform
    _, _, vh = np.linalg.svd(aq, full_matrices=False)
    return q@vh[-min(count, q.shape[1]):].T


def response_basis(u, au):
    """Return B,C with A B=C and C.T C=I, dropping near-null directions."""
    c, s, vt = np.linalg.svd(au, full_matrices=False)
    if not len(s) or s[0] == 0:
        return None, None
    keep = s > 1e-12*s[0]
    return u@vt[keep].T/s[keep], c[:, keep]
