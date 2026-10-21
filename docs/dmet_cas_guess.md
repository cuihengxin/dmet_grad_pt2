# Reusing CAS guesses across DMET geometries

Create one cache outside the geometry loop, and pass the same object through
`solver_option` to each new SSDMET or AODMET gradient object:

```python
from embed_sim.grad.guess import CASGuessCache

cache = CASGuessCache()
options = dict(solver='CASSCF', ncas=6, nelec=6, guess_cache=cache)
# For SA-CASSCF use solver='SA-CASSCF' and the usual statelis/state options.
# NEVPT/NEVPT2 also accept guess_cache for their reference CAS calculation.

for mol in geometries:
    # Run the current geometry's reference calculation and rebuild DMET normally.
    dmet = build_dmet(mol)  # Your existing reference + DMET construction routine.
    grad = dmet.Gradients()
    grad.solver_option = dict(options)
    gradient, energy = grad.kernel()
```

The first call uses `init_guess`/`ci0` if supplied, otherwise existing defaults.
After a successful gradient with a converged CAS solver, the cache saves copies
of the full AO orbitals `[frozen occupied, embedding @ CAS MO, frozen virtual]`
and CI coefficients. The next call reorthogonalizes those AO coefficients at the
new geometry with PySCF `project_init_guess`, projects the active orbitals into
the new embedding, symmetrically orthogonalizes active orbitals, chooses core
orbitals from the retained previous core subspace, and completes virtuals. The
previous CI is passed as `ci0`, overriding first-step guesses when a cache exists.

This is the active-first projection approach, not a fresh AVAS selection. CI is
an approximate warm start in the transported orbital basis; this does not perform
an exact many-electron projection or automatic electronic-state tracking. New
reference/DMET spaces are still built at each geometry. The cache only carries
the embedded CAS initial guess. It follows the most recent successful call,
including trial geometries if used in an optimizer.

Rank loss raises `ValueError` instead of silently replacing active orbitals.
Call `cache.reset()` to deliberately restart from current-geometry guesses.
Reset before changing the molecular system, CAS size, spin settings or SA root
configuration. Use a separate cache for each independent scan. Copies of option
dictionaries must retain the same cache object (avoid deepcopy of the cache).

This checkout lacks the custom `embed_sim.cahf`, `sacasscf_mixer`, and `myavas`
modules imported by the main DMET implementation. The standalone projection and
PySCF CASSCF restart can be tested with `python tests/test_dmet_guess.py`; full
DMET/SA-mixer integration requires the installation containing those modules.

For NEVPT/NEVPT2, cache updates use `grad.es_solver.mc`, the optimized reference
CAS object. They do not mix NEVPT natural orbitals with untransformed reference
CI coefficients. Both the reference `kernel` and `mc2step` accept the transported
`ci0`. `cache.has_guess` can be used to run AVAS only on the first evaluation.
See `examples/a1_cas_pt_opt_ls_cached.py` for a geometry-optimization callback.
