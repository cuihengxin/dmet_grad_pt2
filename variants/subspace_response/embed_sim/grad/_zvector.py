"""Matrix-free Z-vector solution with verified residuals and MO-block transport."""
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from pyscf import gto, lib


def _groups(occ):
    groups = [np.flatnonzero(occ == 2), np.flatnonzero((occ > 0) & (occ < 2)),
              np.flatnonzero(occ == 0)]
    if not np.array_equal(np.concatenate(groups), np.arange(len(occ))):
        raise ValueError('Z-vector requires double/single/virtual MO ordering')
    return groups


def _blocks(occ):
    d, h, v = _groups(occ)
    return [(h, d), (v, d), (v, h)]


class ZVectorCache:
    """Recent solutions, transported separately into the current MO gauge."""
    def __init__(self, max_history=3):
        if not isinstance(max_history, int) or max_history < 1:
            raise ValueError("max_history must be a positive integer")
        self.max_history = max_history
        self.history = []
        self.data = None
        self.recycle_data = None

    def reset(self):
        self.history.clear()
        self.data = None
        self.recycle_data = None

    def update(self, mf, z):
        if self.data is not None and not self._compatible(self.data, mf):
            self.reset()
        self.data = (mf.mol.copy(), mf.mo_coeff.copy(), mf.mo_occ.copy(), z.copy())
        self.history.append(self.data)
        self.history = self.history[-self.max_history:]

    @staticmethod
    def _compatible(data, mf):
        mol, coeff, occ, z = data
        new = mf.mol
        return (mol.natm == new.natm and mol.nao == new.nao and mol.spin == new.spin
                and mol.charge == new.charge and mol.ao_labels() == new.ao_labels()
                and mol._basis == new._basis and np.array_equal(occ, mf.mo_occ))

    def guess(self, mf):
        if self.data is None:
            return None, 'empty'
        return self._transport(self.data, mf)

    def guesses(self, mf):
        return [z for entry in reversed(self.history)
                if (z := self._transport(entry, mf)[0]) is not None]

    def _transport(self, data, mf):
        mol, coeff, occ, z = data
        new = mf.mol
        if not self._compatible(data, mf):
            return None, 'incompatible molecule/basis/occupations'
        # Cross-geometry AO overlap followed by a separate polar alignment
        # inside each occupation class. Never mix occupied and virtual spaces.
        overlap = coeff.T @ gto.intor_cross('int1e_ovlp', mol, new) @ mf.mo_coeff
        rotations = {}
        for idx in _groups(occ):
            if len(idx):
                u, singular, vt = np.linalg.svd(overlap[np.ix_(idx, idx)])
                if singular.min() < .5:
                    return None, 'occupation subspaces changed too much'
                rotations[tuple(idx)] = u @ vt
            else:
                rotations[tuple(idx)] = np.empty((0, 0))
        result, offset = [], 0
        matrix = z[:, None] if z.ndim == 1 else z
        for left, right in _blocks(occ):
            size = len(left)*len(right)
            block = matrix[offset:offset+size].reshape(len(left), len(right), matrix.shape[1])
            mapped = np.einsum('ai,abk,bj->ijk', rotations[tuple(left)],
                               block, rotations[tuple(right)], optimize=True)
            result.append(mapped.reshape(size, matrix.shape[1]))
            offset += size
        result = np.concatenate(result, axis=0)
        return (result[:, 0] if z.ndim == 1 else result), 'transported'

    def update_recycle(self, mf, vectors):
        self.recycle_data = (mf.mol.copy(), mf.mo_coeff.copy(),
                             mf.mo_occ.copy(), vectors.copy())

    def recycled(self, mf):
        if self.recycle_data is None:
            return None
        return self._transport(self.recycle_data, mf)[0]



def solve_zvector(grad, mf, w, gaps, contract, extract, log):
    """Solve (D+K)Z=-W using the original D^-1 left scaling, without shifting D."""
    start = time.perf_counter()
    if np.iscomplexobj(mf.mo_coeff):
        raise NotImplementedError('Z-vector transport currently supports real orbitals')
    _groups(mf.mo_occ)
    if not np.all(np.isfinite(gaps)) or np.any(gaps == 0):
        raise ValueError('Zero/nonfinite orbital gap in Z-vector preconditioner')
    method = grad.zvector_solver.lower()
    if method not in ('gmres', 'krylov'):
        raise ValueError('zvector_solver must be gmres or krylov')
    tol = grad._krylov_conv_tol
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError('Z-vector tolerance must be finite and positive')
    if grad.zvector_restart < 1 or grad.zvector_max_cycle < 1:
        raise ValueError('Z-vector restart and max_cycle must be positive')
    recycling = getattr(grad, 'zvector_recycle', False)
    recycle_dim = getattr(grad, 'zvector_recycle_dim', 8)
    if recycling and (method != 'krylov' or not isinstance(recycle_dim, int) or recycle_dim < 1):
        raise ValueError('Recycling requires Krylov and a positive integer recycle_dim')
    subspace_tol = getattr(grad, "zvector_subspace_tol", None)
    if subspace_tol is None:
        subspace_tol = tol
    if not np.isfinite(subspace_tol) or subspace_tol <= 0:
        raise ValueError("zvector_subspace_tol must be finite and positive")
    calls, iterations = 0, []

    def coupling(z):
        nonlocal calls
        calls += 1
        za = contract(np.asarray(z).ravel())
        return extract(za-za.T, axis=0)/gaps

    def operator(z):
        return z+coupling(z)

    b = -w/gaps
    cache = getattr(grad, 'zvector_cache', None)
    if cache is None:
        cache = getattr(grad.base, '_zvector_cache', None)
        if cache is None:
            cache = grad.base._zvector_cache = ZVectorCache()
    mode = getattr(grad, 'zvector_guess_mode', 'previous')
    if mode not in ('previous', 'history'):
        raise ValueError('zvector_guess_mode must be previous or history')
    initial_residual = b.copy()
    initial = float(np.linalg.norm(b))
    x0, status = None, 'disabled'
    history_count, history_rank, history_weights = 0, 0, []
    setup_calls = calls
    if grad.zvector_warm_start and mode == 'history':
        vectors = cache.guesses(mf)
        history_count = len(vectors)
        status = 'empty'
        if vectors:
            v = np.column_stack(vectors)
            y = np.column_stack([operator(z) for z in vectors])
            latest_residual = float(np.linalg.norm(b-y[:, 0]))
            if np.isfinite(latest_residual) and latest_residual < initial:
                x0, initial, status = vectors[0], latest_residual, 'history: latest fallback'
                initial_residual = b-y[:, 0]
            norms = np.linalg.norm(y, axis=0)
            keep = np.isfinite(norms) & (norms > 1e-30)
            if np.any(keep) and np.all(np.isfinite(y[:, keep])):
                # Scale columns before truncated-SVD least squares. Dropping
                # small singular directions prevents extrapolating noisy history.
                c, _, history_rank, _ = np.linalg.lstsq(
                    y[:, keep]/norms[keep], b, rcond=1e-8)
                weights = c/norms[keep]
                predicted = float(np.linalg.norm(b-y[:, keep]@weights))
                if np.isfinite(predicted) and predicted < initial:
                    trial_x = v[:, keep]@weights
                    # Confirm cancellation-prone combinations with a fresh action.
                    trial_residual = b-operator(trial_x)
                    trial = float(np.linalg.norm(trial_residual))
                    if np.isfinite(trial) and trial < initial:
                        x0, initial, status = trial_x, trial, 'history: least squares'
                        initial_residual = trial_residual
                        history_weights = weights.tolist()
    elif grad.zvector_warm_start:
        x0, status = cache.guess(mf)
        if x0 is not None:
            trial_residual = b-operator(x0)
            trial = float(np.linalg.norm(trial_residual))
            if np.isfinite(trial) and trial < initial:
                initial = trial
                initial_residual = trial_residual
            else:
                x0, status = None, 'rejected: residual not smaller than zero guess'
    recycle_u, recycle_au, coarse_b, coarse_c = None, None, None, None
    recycle_calls = calls
    if recycling:
        recycle_u = cache.recycled(mf)
        if recycle_u is not None and recycle_u.shape[1]:
            from embed_sim.grad._zvector_recycle import response_basis
            recycle_au = np.column_stack([operator(u) for u in recycle_u.T])
            coarse_b, coarse_c = response_basis(recycle_u, recycle_au)
            if coarse_b is not None and coarse_b.shape[1]:
                weights = coarse_c.T@initial_residual
                update = coarse_b@weights
                x0 = update if x0 is None else x0+update
                initial_residual = initial_residual-coarse_c@weights
                initial = float(np.linalg.norm(initial_residual))
                status += ' + recycled'
    recycle_calls = calls-recycle_calls
    active_recycle = coarse_b is not None and coarse_b.shape[1] > 0
    raw_basis, raw_images = [], []
    actual_images = []

    def projected_coupling(v):
        v = np.asarray(v).ravel()
        av = operator(v)
        actual_images.append(av.copy())
        return av-coarse_c@(coarse_c.T@av)-v

    def record_basis(cycle, xs, ax):
        iterations.append(cycle)
        if recycling:
            raw_basis[:] = xs
            raw_images[:] = actual_images if active_recycle else [x+a for x, a in zip(xs, ax)]

    guess_matvecs = calls-setup_calls
    correction = (getattr(grad, 'zvector_reuse_residual', True) or active_recycle) and x0 is not None
    solve_b = initial_residual if correction else b
    solve_x0 = None if correction else x0
    if b.size == 0:
        z = b.copy()
    elif method == 'gmres':
        # callback_type=legacy makes maxiter count INNER iterations in SciPy.
        a = LinearOperator((b.size, b.size), matvec=operator, dtype=b.dtype)
        z, info = gmres(a, solve_b, x0=solve_x0, rtol=0., atol=tol,
                       restart=min(grad.zvector_restart, b.size),
                       maxiter=grad.zvector_max_cycle,
                       callback=lambda r: iterations.append(float(r)), callback_type='legacy')
        if info != 0:
            raise RuntimeError(f'Z-vector GMRES failed: info={info}')
    else:
        z = lib.krylov(projected_coupling if active_recycle else coupling, solve_b, x0=solve_x0, tol=subspace_tol,
                       max_cycle=grad.zvector_max_cycle, verbose=grad.verbose,
                       callback=record_basis,
                       lindep=min(lib.linalg_helper.DSOLVE_LINDEP, subspace_tol**2))
    if active_recycle:
        z = z-coarse_b@(coarse_c.T@operator(z))
    if correction:
        z = z+x0
    residual = operator(z)-b
    scaled = float(np.linalg.norm(residual))
    raw = float(np.linalg.norm(gaps*residual))
    # Legacy Krylov may terminate by linear dependence rather than true residual.
    if not np.isfinite(scaled) or scaled > max(10*tol, 1e-12):
        raise RuntimeError(f'Z-vector true scaled residual {scaled:.3g} exceeds tolerance')
    cache.update(mf, z)
    extraction_start = time.perf_counter()
    stored_dim = 0
    if recycling and len(raw_basis) >= getattr(grad, 'zvector_recycle_min_iterations', 12):
        from embed_sim.grad._zvector_recycle import slow_directions
        if recycle_u is not None:
            raw_basis = list(recycle_u.T)+list(raw_basis)
            raw_images = list(recycle_au.T)+list(raw_images)
        directions = slow_directions(raw_basis, raw_images, recycle_dim)
        if directions is not None:
            cache.update_recycle(mf, directions)
            stored_dim = directions.shape[1]
    extraction_seconds = time.perf_counter()-extraction_start
    grad.zvector_stats = dict(solver=method, warm_start=status,
        initial_scaled_residual=initial, scaled_residual=scaled, raw_residual=raw,
        residual_reused=correction, recycle_active=bool(active_recycle),
        recycle_setup_matvecs=recycle_calls, recycle_stored_dim=stored_dim,
        recycle_extraction_seconds=extraction_seconds,
        contract_stats=dict(getattr(contract, 'stats', {})),
        guess_mode=mode, history_count=history_count, history_rank=int(history_rank),
        history_weights=history_weights, guess_matvecs=guess_matvecs,
        relative_raw_residual=raw/max(float(np.linalg.norm(w)), 1e-300),
        matvecs=calls, iterations=len(iterations),
        seconds=time.perf_counter()-start)
    log.info('Z-vector %s: %d matrix actions; residual %.3g; guess %s',
             method, calls, scaled, status)
    return z
