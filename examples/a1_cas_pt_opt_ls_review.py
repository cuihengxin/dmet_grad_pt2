"""a1 LS: ROHF-X2C / AODMET / three-doublet SA-CASSCF / root-0 NEVPT2.

Run from a NEW output directory:
  python /path/to/examples/a1_cas_pt_opt_ls_review.py > run.log 2>&1

This follows the lowest SA-CASSCF root index, NOT a diabatic state identified
by wavefunction overlap. Diagnostics flag possible discontinuities; they do
not prove state continuity. All saved evaluations include optimizer trials.
The script deliberately stops on a bath dimension change or failed CAS.
No automatic restart or automatic root switching is performed.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
from pyscf import gto, scf
from pyscf.mcscf.chkfile import dump_mcscf
from pyscf.geomopt import as_pyscf_method, geometric_solver

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embed_sim import aodmet, sacasscf_mixer
from embed_sim.grad import aodmet as aodmet_grad  # registers Gradients
from embed_sim.grad.guess import CASGuessCache


# ---------------- Editable input ----------------
XYZ = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/sse17_struc/a1_2.xyz'
INITIAL_CHK = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/05a1/clustera1_rohf_r_M_cas_full_rohf.chk'
PREFIX = 'a1_ls_review'
BASIS = {'default': 'ccpvdz-dk', 'Fe': 'ccpwCVTZ-dk',
         'O': 'ccpvtz-dk', 'N': 'ccpvtz-dk'}
CHARGE, SPIN = 1, 1                 # PySCF spin = Nalpha - Nbeta
NCAS, NELEC = 10, 9
STATES = [0, 3, 0, 0, 0, 0]        # three doublets; mixer default weights
STATE = 0                         # root index, not overlap-followed state
ACTIVE_LABELS = ['Fe 3d', 'Fe 4d']
IMPURITY_LABELS = ['Fe.*', 'O.*p', 'N.*p']
MAX_MEMORY = 180000                # MB; set to actual allocated memory
SCF_TOL = 1e-10
CAS_TOL = 1e-9
Z_TOL = 1e-9
BATH_THRESHOLD = 1e-12             # retain original model; test separately
STRICT_BATH = True
TRUST, MAX_TRUST = 0.05, 0.10       # Angstrom
MAX_GEOM_STEPS = 100
# ------------------------------------------------


def require_cas(mc):
    if not np.all(getattr(mc, 'converged', False)):
        raise RuntimeError('SA-CASSCF failed: no gradient may be accepted.')


def checked_reference_kernel(mc):
    """Stop before NEVPT2 if its reference CAS did not converge.

    The current NEVPT entry constructs a fresh mixer for each evaluation.
    Install this wrapper only for this standalone process.
    """
    original = mc.kernel

    def kernel(*args, **kwargs):
        result = original(*args, **kwargs)
        require_cas(mc)
        return result

    mc.kernel = kernel
    return mc


def main():
    # Fail before overwriting any previous run artifacts.
    out = Path(PREFIX)
    out.mkdir(exist_ok=False)
    mol = gto.M(atom=XYZ, basis=BASIS, charge=CHARGE, spin=SPIN,
                max_memory=MAX_MEMORY, verbose=4)
    mol.tofile(str(out / 'initial.xyz'), format='xyz')

    mf = scf.ROHF(mol).x2c()
    mf.conv_tol = SCF_TOL
    mf.max_cycle = 250
    mf.chkfile = str(out / 'reference.chk')
    # Checkpoint projection uses its stored geometry, rather than copying AO
    # coefficient arrays between possibly different structures.
    dm = mf.init_guess_by_chkfile(INITIAL_CHK, project=True)
    # Shift only for initial stabilization; every returned reference is
    # reconverged with zero shift. Subsequent geometries start at zero shift.
    for shift in (1.0, 0.2, 0.0):
        mf.level_shift = shift
        mf.kernel(dm0=dm)
        dm = mf.make_rdm1()
    if not mf.converged:
        raise RuntimeError('Initial zero-shift ROHF failed.')
    scanner = mf.as_scanner()
    imp = mol.search_ao_label(IMPURITY_LABELS)
    cache = CASGuessCache()
    options = dict(solver='NEVPT2', ncas=NCAS, nelec=NELEC,
                   statelis=STATES, state=STATE, conv_tol=CAS_TOL,
                   max_cycle=100, guess_cache=cache)

    def build():
        dmet = aodmet.AODMET(scanner, title=str(out / 'embedding'),
                            imp_idx=imp, threshold=BATH_THRESHOLD, verbose=4)
        dmet.build(save_chk=False)
        return dmet

    first = build()
    ncas, nelec, mo = first.avas(ACTIVE_LABELS, minao=mol._basis['Fe'],
                                threshold=0.5, openshell_option=2)
    if (ncas, nelec) != (NCAS, NELEC):
        raise RuntimeError(f'AVAS selected ({nelec}, {ncas}); expected (9, 10).')
    factory = sacasscf_mixer.sacasscf_mixer
    initial_cas = factory(first.es_mf, NCAS, NELEC, STATES)
    initial_cas.conv_tol = CAS_TOL
    initial_cas.max_cycle = 100
    initial_cas.ah_level_shift = 1e-4
    initial_cas.kernel(mo)
    require_cas(initial_cas)
    cache.update(first, SimpleNamespace(mc=initial_cas), options)

    def guarded_factory(*args, **kwargs):
        mc = factory(*args, **kwargs)
        mc.ah_level_shift = 1e-4
        return checked_reference_kernel(mc)

    sacasscf_mixer.sacasscf_mixer = guarded_factory
    dimensions = (first.nes - len(imp), first.nfo, first.nfv)
    initial_active = (first.es_orb @ initial_cas.mo_coeff)[
        :, initial_cas.ncore:initial_cas.ncore + NCAS].copy()
    previous_active = initial_active.copy()
    previous_mol = mol.copy()
    evaluation = 0

    def evaluate(current):
        nonlocal evaluation, previous_active, previous_mol
        scanner(current)
        if not scanner.converged:
            raise RuntimeError('ROHF failed at trial geometry; inspect before retrying.')
        dmet = build()
        dims = (dmet.nes - len(imp), dmet.nfo, dmet.nfv)
        ldm, _, _ = dmet.lowdin_orth()
        env = np.setdiff1d(np.arange(current.nao), imp)
        occupations = np.linalg.eigvalsh(ldm[np.ix_(env, env)])
        np.save(out / f'bath_{evaluation + 1:04d}.npy', occupations)
        if STRICT_BATH and dims != dimensions:
            current.tofile(str(out / 'bath_change.xyz'), format='xyz')
            raise RuntimeError(f'Bath dimensions changed {dimensions} -> {dims}. '
                               'Inspect occupations; do not silently continue.')

        grad = dmet.Gradients()
        grad._krylov_conv_tol = Z_TOL
        grad.solver_option = dict(options)
        de, energy = grad.kernel()
        mc = grad.es_solver.mc
        require_cas(mc)
        if not np.isfinite(energy) or not np.all(np.isfinite(de)):
            raise RuntimeError('Non-finite energy or gradient.')
        active = (dmet.es_orb @ mc.mo_coeff)[:, mc.ncore:mc.ncore + NCAS]
        # Singular values measure subspace continuity, not many-electron overlap.
        cross = gto.intor_cross('int1e_ovlp', previous_mol, current)
        sv_previous = np.linalg.svd(previous_active.T @ cross @ active,
                                    compute_uv=False)
        cross0 = gto.intor_cross('int1e_ovlp', mol, current)
        sv_initial = np.linalg.svd(initial_active.T @ cross0 @ active,
                                   compute_uv=False)
        rdm1 = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
        natural_occ = np.linalg.eigvalsh(rdm1)[::-1]
        roots = np.asarray(mc.fcisolver.e_states, dtype=float)
        evaluation += 1
        record = dict(evaluation=evaluation, energy=float(energy),
                      scf_energy=float(scanner.e_tot),
                      gradient_max=float(np.abs(de).max()),
                      gradient_rms=float(np.sqrt(np.mean(de**2))),
                      bath_dimensions=list(map(int, dims)),
                      cas_root_energies=roots.tolist(),
                      sa_natural_occupations=natural_occ.tolist(),
                      active_singular_values_previous=sv_previous.tolist(),
                      active_singular_values_initial=sv_initial.tolist())
        with (out / 'diagnostics.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        with (out / 'evaluations.xyz').open('a') as handle:
            handle.write(current.tostring(format='xyz') + '\n')
        np.savez(out / f'evaluation_{evaluation:04d}.npz',
                 coords_bohr=current.atom_coords(), energy=energy, gradient=de)
        # CAS checkpoint + AO embedding matrices preserve inspectable restart
        # ingredients. Reconstruct a new calculation explicitly when restarting;
        # these files do not contain optimizer history or a serialized cache.
        mc.chkfile = str(out / f'cas_{evaluation:04d}.chk')
        dump_mcscf(mc, chkfile=mc.chkfile, ci_vector=mc.ci)
        np.savez(out / f'embedding_{evaluation:04d}.npz',
                 es_orb=dmet.es_orb, fo_orb=dmet.fo_orb, fv_orb=dmet.fv_orb)
        print('DIAGNOSTICS', json.dumps(record), flush=True)
        previous_active, previous_mol = active.copy(), current.copy()
        return energy, de

    try:
        method = as_pyscf_method(mol, evaluate)
        converged, optimized = geometric_solver.kernel(
            method, maxsteps=MAX_GEOM_STEPS, trust=TRUST, tmax=MAX_TRUST,
            convergence_energy=1e-6, convergence_grms=3e-4,
            convergence_gmax=4.5e-4, convergence_drms=1.2e-3,
            convergence_dmax=1.8e-3)
        optimized.tofile(str(out / 'final.xyz'), format='xyz')
        if not converged:
            raise RuntimeError('Geometry optimization did not converge; final.xyz is unfinished.')
        # Same electronic model, explicit final evaluation; no DF/AVAS change.
        energy, gradient = evaluate(optimized)
        print('FINAL', energy, 'max |gradient|', np.abs(gradient).max(), flush=True)
    finally:
        sacasscf_mixer.sacasscf_mixer = factory


if __name__ == '__main__':
    main()
