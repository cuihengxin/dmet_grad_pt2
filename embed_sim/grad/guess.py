"""Explicit, reusable CAS initial guesses for successive DMET geometries.

Keep one CASGuessCache in solver_option['guess_cache'] across gradient objects.
Only successful gradient calculations with converged CAS solvers update it.
"""
from copy import deepcopy
import numpy as np


CAS_SOLVERS = ('CASSCF', 'SA-CASSCF', 'NEVPT', 'NEVPT2')


def _signature(options):
    return (options['solver'], options['ncas'], repr(options['nelec']),
            repr(options.get('statelis')), repr(options.get('weights')),
            options.get('fix_spin', True))


def embedding_guess(basis, overlap, full_mo, total_ncore, ncore, ncas,
                    reference, tol=1e-10):
    """Project active first, then core, completing an orthonormal ES basis.

    Symmetric active orthogonalization preserves the reference orbital gauge.
    Rank loss of the projected active space is an error, never silently repaired.
    """
    metric = basis.conj().T @ overlap @ basis
    w, v = np.linalg.eigh(metric)
    if np.min(w) <= tol:
        raise ValueError('New embedding metric is singular')
    sqrt_s = (v * np.sqrt(w)) @ v.conj().T
    inv_sqrt_s = (v / np.sqrt(w)) @ v.conj().T
    targets = inv_sqrt_s @ basis.conj().T @ overlap @ full_mo
    active = targets[:, total_ncore:total_ncore+ncas]
    u, s, vh = np.linalg.svd(active, full_matrices=False)
    if len(s) != ncas or np.min(s) <= tol:
        raise ValueError('Previous CAS active space loses rank in new embedding')
    active = u @ vh
    # Select the most retained directions of the previous occupied core space.
    core_target = targets[:, :total_ncore]
    core_target = core_target - active @ (active.conj().T @ core_target)
    u, s, _ = np.linalg.svd(core_target, full_matrices=False)
    if ncore and (len(s) < ncore or s[ncore-1] <= tol):
        raise ValueError('Previous CAS core space loses rank in new embedding')
    core = u[:, :ncore]
    occupied = np.column_stack((core, active))
    # Complete with new reference MOs, removing occupied components.
    virtual = sqrt_s @ reference
    virtual -= occupied @ (occupied.conj().T @ virtual)
    u, s, _ = np.linalg.svd(virtual, full_matrices=False)
    nvirt = basis.shape[1] - ncore - ncas
    if nvirt < 0 or (nvirt and (len(s) < nvirt or s[nvirt-1] <= tol)):
        raise ValueError('Cannot complete embedded CAS orbitals')
    return inv_sqrt_s @ np.column_stack((occupied, u[:, :nvirt]))


class CASGuessCache:
    """Warm starts for CAS and NEVPT reference CAS; reset explicitly for a new scan.

    This follows the standard PySCF scanner convention: previous AO coefficient
    arrays are reinterpreted at the new geometry, then reorthogonalized. CI is
    reused as an approximate guess, not an exact many-electron projection.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._snapshot = None

    @property
    def has_guess(self):
        return self._snapshot is not None

    def update(self, dmet, solver, options):
        if options['solver'] not in CAS_SOLVERS:
            raise ValueError('CASGuessCache supports CASSCF, SA-CASSCF, NEVPT and NEVPT2 only')
        if options['solver'] in ('NEVPT', 'NEVPT2'):
            # Cache the optimized reference, before NEVPT natural-orbital changes.
            # Keep MO and CI from the same object and orbital gauge.
            solver = solver.mc
        if not np.all(getattr(solver, 'converged', False)):
            return
        # Equivalent to total_cas(), without requiring the SA mixer for SS CAS.
        full_mo = np.column_stack((dmet.fo_orb,
                                   dmet.es_orb @ solver.mo_coeff,
                                   dmet.fv_orb))
        self._snapshot = dict(mo=full_mo.copy(), ci=deepcopy(solver.ci),
                              mol=dmet.mol.copy(), signature=_signature(options),
                              total_ncore=dmet.nfo + solver.ncore,
                              ncas=solver.ncas, nelec=solver.nelecas)

    def prepare(self, dmet, options):
        from pyscf import mcscf, gto
        result = dict(options)
        old = self._snapshot
        if old is None:
            return result
        if _signature(options) != old['signature']:
            raise ValueError('CAS settings changed: reset guess_cache first')
        if (not gto.same_basis_set(old['mol'], dmet.mol)
                or not np.array_equal(old['mol'].atom_charges(), dmet.mol.atom_charges())
                or old['mol'].spin != dmet.mol.spin
                or old['mol'].nelectron != dmet.mol.nelectron):
            raise ValueError('Molecular system changed: reset guess_cache first')
        total = mcscf.CASSCF(dmet.mf_or_cas, old['ncas'], old['nelec'])
        full_mo = mcscf.addons.project_init_guess(total, old['mo'])
        # The total core count stays fixed; the embedded count can change with nfo.
        ncore = total.ncore - dmet.nfo
        if ncore < 0:
            raise ValueError('New frozen core exceeds total CAS core')
        result['init_guess'] = embedding_guess(
            dmet.es_orb, dmet.mf_or_cas.get_ovlp(), full_mo,
            total.ncore, ncore, old['ncas'], dmet.es_mf.mo_coeff)
        result['ci0'] = deepcopy(old['ci'])
        return result


def prepare_options(dmet, options):
    cache = options.get('guess_cache')
    if cache is None:
        return options
    if options['solver'] not in CAS_SOLVERS:
        raise ValueError('guess_cache supports CASSCF, SA-CASSCF, NEVPT and NEVPT2 only')
    return cache.prepare(dmet, options)
