"""Reproducible spectral, timing, and molecular gradient comparison (one thread)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import time
from types import SimpleNamespace
import numpy as np
from pyscf import lib, gto, scf
from embed_sim.grad._embedding_response import embedding_response_factors
from embed_sim.grad._embedding_response_legacy import aodmet_response_factors


class Log:
    def debug1(self, *args):
        pass


def legacy(occ, nv, nb):
    return aodmet_response_factors(SimpleNamespace(_perturb_anyway=False,
                                                  _perturbation=1e-10), occ, nv, nb, Log())


def spectral_checks():
    rng = np.random.default_rng(22)
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    e = rng.normal(size=(8, 8)); e = (e + e.T) / 2
    occ = np.array([0., 0., .4, .4, .4, .4, 2., 2.])
    h = q @ np.diag(occ) @ q.T
    perturb = q.T @ e @ q
    bath = q[:, 2:6]

    def derivative(factors, vectors=q, coupling=perturb):
        db = vectors @ (factors * coupling[2:6]).T
        b = vectors[:, 2:6]
        return db @ b.T + b @ db.T

    new = embedding_response_factors(occ, 2, 4)[0]
    old = legacy(occ, 2, 4)[0]
    analytic = derivative(new)
    errors = []
    for step in (1e-4, 5e-5, 1e-5):
        def projector(sign):
            _, u = np.linalg.eigh(h + sign * step * e)
            return u[:, 2:6] @ u[:, 2:6].T
        fd = (projector(1) - projector(-1)) / (2 * step)
        errors.append(float(np.max(np.abs(fd - analytic))))
    assert errors[-1] < 1e-7
    r, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    rotated = q.copy(); rotated[:, 2:6] = bath @ r
    gauge_error = float(np.max(np.abs(analytic - derivative(new, rotated, rotated.T @ e @ rotated))))
    assert gauge_error < 1e-12
    # Near degeneracy inside bath is projected even when individual eigenvector
    # derivatives are huge; external response is not globally damped.
    near = np.array([0., 1e-6, 1.001e-6, .7, 2.])
    nf = embedding_response_factors(near, 1, 3)[0]
    of = legacy(near, 1, 3)[0]
    assert np.all(nf[:, 1:4] == 0)
    assert nf[0, 0] == 1e6
    assert of[0, 0] < 1e4
    with np.testing.assert_raises(ValueError):
        embedding_response_factors([0., 0., 2.], 1, 1)
    damp = embedding_response_factors([0., 0., 2.], 1, 1, 1e-10)
    assert np.all(np.isfinite(damp[0]))
    np.testing.assert_allclose(embedding_response_factors([0., .4, .8, 2.], 1, 2)[0][:, [0, 3]],
                               legacy(np.array([0., .4, .8, 2.]), 1, 2)[0][:, [0, 3]])
    timings = {}
    large = np.r_[np.zeros(200), np.repeat(np.linspace(.1, 1.9, 25), 4), np.full(200, 2.)]
    for name, fn in [('subspace', embedding_response_factors), ('legacy', legacy)]:
        samples = []
        for repeat in range(5):
            start = time.perf_counter()
            for _ in range(50):
                fn(large, 200, 100)
            samples.append((time.perf_counter() - start) / 50)
        timings[name] = float(np.median(samples))
    return dict(projector_fd_max_errors=errors, gauge_max_error=gauge_error,
                exact_degenerate_old_new_max_difference=float(np.max(np.abs(analytic - derivative(old)))),
                near_boundary_external_factor_new=float(nf[0, 0]),
                near_boundary_external_factor_old=float(of[0, 0]),
                seconds_per_factor_call_500env_100bath=timings)


def molecular_checks(square=False):
    from embed_sim import aodmet, ssdmet
    records = []
    for module in (aodmet, ssdmet):
        # Four well separated identical H2 fragments: four equal bath occupations.
        def evaluate(delta=0., mode='subspace', direction=None):
            coords = np.array([[0., 1000.*i, z] for z in (0., 1.4) for i in range(4)])
            if square:
                coords = np.array([[0., 0., 0.], [0., 2., 2.], [0., 0., 2.], [0., 2., 0.]])
            if direction is None:
                coords[0, 2] += delta
            else:
                coords += delta * direction
            mol = gto.M(atom=[('H', xyz) for xyz in coords], unit='Bohr', basis='6-31g', verbose=0)
            mf = scf.ROHF(mol).run(conv_tol=1e-12)
            assert mf.converged
            cls = module.AODMET if module is aodmet else module.SSDMET
            dmet = cls(mf, imp_idx=list(range(4 if square else 8)), threshold=1e-10, verbose=0)
            dmet.build(save_chk=False)
            grad = dmet.Gradients()
            grad.embedding_response = mode
            grad.solver_option = dict(solver='ROHF')
            grad._krylov_conv_tol = 1e-10
            t = time.perf_counter()
            result, energy = grad.kernel()
            elapsed = time.perf_counter() - t
            ldm = dmet.lowdin_orth()[0]
            idx = [i for i in range(mol.nao) if i not in dmet.imp_idx]
            occ = np.linalg.eigvalsh(ldm[np.ix_(idx, idx)])
            return float(energy), result, elapsed, occ.tolist()
        en, gn, tn, occ = evaluate()
        eo, go, to, _ = evaluate(mode='legacy')
        fd = []
        for step in (1e-3, 5e-4):
            ep = evaluate(step)[0]; em = evaluate(-step)[0]
            numeric = (ep - em) / (2 * step)
            fd.append(dict(step=step, numerical=numeric,
                           subspace_error=float(abs(numeric - gn[0, 2])),
                           legacy_error=float(abs(numeric - go[0, 2]))))
        directional_fd = []
        if not square:
            direction = np.zeros_like(gn)
            direction[4, 2] = .7
            direction[1, 2] = -.4
            direction[5, 2] = .2
            analytic = float(np.sum(gn * direction))
            for step in (1e-3, 5e-4):
                numeric = (evaluate(step, direction=direction)[0] -
                           evaluate(-step, direction=direction)[0]) / (2 * step)
                directional_fd.append(dict(step=step, analytic=analytic, numerical=numeric,
                                           error=abs(numeric - analytic)))
            assert directional_fd[-1]['error'] < 1e-6
            assert directional_fd[-1]['error'] < .5 * directional_fd[0]['error'] + 1e-9
            assert fd[-1]['subspace_error'] < .5 * fd[0]['subspace_error'] + 1e-9
            assert np.isfinite(gn).all() and np.isfinite(go).all()
            assert abs(en - eo) < 1e-10
            np.testing.assert_allclose(gn.sum(axis=0), 0., atol=1e-7)
            bath_occ = np.asarray(occ)[np.asarray(occ) > 1e-10]
            assert len(bath_occ) == 4 and np.ptp(bath_occ) < 1e-8
            assert fd[-1]["subspace_error"] < 1e-6
            assert np.max(abs(gn-go)) < 1e-8
        print(module.__name__, "FD", fd, "occupations", occ, flush=True)
        records.append(dict(implementation=module.__name__, occupations=occ,
                            energy=en, legacy_energy=eo, gradient=gn.tolist(),
                            legacy_gradient=go.tolist(), gradient_max_difference=float(np.max(abs(gn-go))),
                            kernel_seconds=dict(subspace=tn, legacy=to), finite_differences=fd,
                            mixed_direction_finite_differences=directional_fd))
    return records


if __name__ == '__main__':
    lib.num_threads(1)
    square = '--square-diagnostic' in sys.argv
    result = dict(spectral=spectral_checks(), molecular=molecular_checks(square))
    target = Path('tests/embedding_response_square_diagnostic.json' if square else 'tests/embedding_response_results.json')
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
