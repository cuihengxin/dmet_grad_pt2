"""One-thread real molecular gradient benchmark; JSON includes every timed sample."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import argparse
import time
import numpy as np
from pyscf import gto, scf, lib
from embed_sim import aodmet, ssdmet
from embed_sim.grad import aodmet as _aodmet_grad, ssdmet as _ssdmet_grad
from embed_sim.grad._zvector import ZVectorCache

lib.num_threads(1)


def evaluate(coords, module, solver, warm, cache, spin=0, guess_mode="previous", cache_static=False, reuse_residual=False,
             recycle=False, recycle_dim=8, recycle_min_iterations=12):
    mol = gto.M(atom=[('H', r) for r in coords], basis='6-31g', unit='Bohr', spin=spin, verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-12)
    assert mf.converged
    cls = module.AODMET if module is aodmet else module.SSDMET
    dmet = cls(mf, imp_idx=list(range(4)), threshold=1e-10, verbose=0)
    dmet.build(save_chk=False)
    grad = dmet.Gradients()
    grad.solver_option = dict(solver='CASSCF', ncas=dmet.es_mf.mo_coeff.shape[1],
                              nelec=dmet.es_mf.mol.nelectron, fix_spin=False,
                              conv_tol=1e-12, max_cycle=100)
    grad.zvector_solver, grad.zvector_warm_start = solver, warm
    grad.zvector_cache = cache
    grad.zvector_guess_mode = guess_mode
    grad.zvector_cache_static = cache_static
    grad.zvector_reuse_residual = reuse_residual
    grad.zvector_recycle = recycle
    grad.zvector_recycle_dim = recycle_dim
    grad.zvector_recycle_min_iterations = recycle_min_iterations
    grad._krylov_conv_tol = 1e-8
    start = time.perf_counter()
    g, energy = grad.kernel()
    return dict(gradient_seconds=time.perf_counter()-start, zvector=grad.zvector_stats,
                energy=float(energy), gradient=g.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--krylov-only', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    configs = ([('krylov', False), ('krylov', True)] if args.krylov_only else
               [('krylov', False), ('krylov', True), ('gmres', False), ('gmres', True)])
    records = []
    for module in (aodmet, ssdmet):
        for spin in (0, 1):
            coords = np.array([[0., 0., 0.], [0., 0., 1.4],
                               [0., 4., .2], [.2, 4., 1.8]])
            if spin:
                coords = np.vstack([coords, [1.0, 2.0, 3.5]])
            next_coords = coords.copy(); next_coords[0, 2] += .001
            for repeat in range(args.repeats):
                # Alternate ordering to reduce systematic timing bias.
                for solver, warm in (configs if repeat % 2 == 0 else configs[::-1]):
                    cache = ZVectorCache()
                    first = evaluate(coords, module, solver, warm, cache, spin)
                    second = evaluate(next_coords, module, solver, warm, cache, spin)
                    record = dict(module=module.__name__, spin=spin, repeat=repeat,
                                  solver=solver, warm=warm, first=first, second=second)
                    records.append(record)
                    print(module.__name__, spin, repeat, solver, warm,
                          second['gradient_seconds'], second['zvector'], flush=True)
            relevant = [r for r in records if r['module'] == module.__name__ and r['spin'] == spin]
            ref = relevant[0]['second']
            for record in relevant:
                np.testing.assert_allclose(record['second']['gradient'], ref['gradient'], atol=2e-8, rtol=0)
                assert abs(record['second']['energy']-ref['energy']) < 1e-10
            # Independent energy finite difference at the second geometry.
            fd = []
            for h in (.001, .0005):
                plus, minus = next_coords.copy(), next_coords.copy()
                plus[0, 2] += h; minus[0, 2] -= h
                ep = evaluate(plus, module, 'krylov' if args.krylov_only else 'gmres', False, ZVectorCache(), spin)['energy']
                em = evaluate(minus, module, 'krylov' if args.krylov_only else 'gmres', False, ZVectorCache(), spin)['energy']
                fd.append(dict(step=h, numerical=(ep-em)/(2*h),
                               error=abs((ep-em)/(2*h)-ref['gradient'][0][2])))
            print('FD', module.__name__, spin, fd, flush=True)
            assert fd[-1]['error'] < 2e-6, fd
            for r in relevant:
                r['finite_difference'] = fd
    path = Path(__file__).with_name('zvector_krylov_warm_results.json' if args.krylov_only else 'zvector_benchmark_results.json')
    path.write_text(json.dumps(dict(threads=1, basis='6-31g', embedded_solver='full-active CASSCF (FCI)', unit='Bohr', records=records), indent=2))

if __name__ == '__main__':
    main()
