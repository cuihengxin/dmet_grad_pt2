"""Paired DMET nuclear-gradient benchmark; local modules always take priority.

Run with --dependency-path pointing at an existing complete embed_sim directory.
Only missing support modules are loaded there. Timings use one BLAS/OpenMP thread.
"""
import argparse
import importlib
import hashlib
import platform
import json
from pathlib import Path
import sys
import time
from unittest import mock

import numpy as np
import pyscf
from pyscf import gto, scf, lib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(args):
    import embed_sim
    if args.dependency_path:
        embed_sim.__path__.append(str(Path(args.dependency_path).resolve()))
    from embed_sim import ssdmet
    from embed_sim.grad import ssdmet as outer, solvers
    from embed_sim.dmet_nevpt.grad.gradients import Gradients
    from embed_sim.fic_nevpt.grad import ci_contraction
    lib.num_threads(1)
    mol = gto.M(atom=';'.join('H 0 0 %s' % (args.spacing*i + 0.15*(i % 2)) for i in range(args.atoms)),
                basis='sto-3g', unit='Angstrom', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    assert mf.converged
    dmet = ssdmet.SSDMET(mf, imp_idx=list(range(args.impurity)),
                         threshold=1e-10, verbose=0)
    dmet.build(save_chk=False)
    # Outer DMET code expects spin-resolved densities in its ROHF response.
    outer_mf = scf.ROHF(mol)
    outer_mf.__dict__.update(mf.__dict__)
    dmet.mf_or_cas = outer_mf
    options = dict(solver='NEVPT', ncas=args.ncas, nelec=args.ncas,
                   statelis=[2], state=0, canonstep=args.canonstep, conv_tol=1e-11, max_cycle=100)
    # A two-root reference avoids single-root container compatibility adapters.
    mc = solvers.sacasscf_mixer.sacasscf_mixer(
        dmet.es_mf, args.ncas, args.ncas, [2])
    if args.optimizer == 'newton':
        mc = mc.newton()
    mc.conv_tol = 1e-11
    mc.conv_tol_grad = 1e-6
    mc.max_cycle_macro = 300
    mc.kernel()
    assert mc.converged, 'CASSCF not converged'
    samples = []
    for repeat in range(args.repeats + 1):
        for backend in (('legacy', 'contracted') if repeat % 2 == 0
                        else ('contracted', 'legacy')):
            captured = {}
            source_method = Gradients.get_ci_response
            solve_method = Gradients.solve_lagrange
            def source(grad):
                value = source_method(grad)
                captured['source'] = value.copy()
                captured['stats'] = dict(grad.ci_response_stats)
                return value
            def solve(grad, *a, **kw):
                value = solve_method(grad, *a, **kw)
                captured['converged'] = bool(value[0])
                captured['zvector'] = value[1].copy()
                captured['rhs'] = value[2].copy()
                return value
            inner_method = outer.get_es_imds
            def inner(*a, **kw):
                start = time.perf_counter()
                value = inner_method(*a, **kw)
                captured['inner_seconds'] = time.perf_counter() - start
                captured['rdm1'], captured['rdm2'], captured['X'] = value[2:]
                return value
            grad = dmet.Gradients()
            grad.solver_option = dict(options, ci_response_backend=backend)
            # Reuse exactly the same converged CASSCF; PT/gradient objects are fresh.
            with mock.patch.object(solvers.sacasscf_mixer, 'sacasscf_mixer', return_value=mc), \
                 mock.patch.object(mc, 'kernel'), \
                 mock.patch.object(Gradients, 'get_ci_response', source), \
                 mock.patch.object(Gradients, 'solve_lagrange', solve), \
                 mock.patch.object(outer, 'get_es_imds', inner):
                start = time.perf_counter()
                nuclear, energy = grad.kernel()
                seconds = time.perf_counter() - start
            assert captured['converged'], 'inner Z-vector not converged'
            sample = dict(backend=backend, repeat=repeat, seconds=seconds,
                          energy=float(energy), nuclear=nuclear.tolist())
            sample.update({k: v.tolist() if isinstance(v, np.ndarray) else v
                           for k, v in captured.items()})
            samples.append(sample)
            print(backend, repeat, 'total', seconds, 'source', captured['stats']['seconds'], flush=True)
    paired = []
    for repeat in range(args.repeats + 1):
        old, new = [next(s for s in samples if s['repeat'] == repeat and s['backend'] == b)
                    for b in ('legacy', 'contracted')]
        paired.append({key: float(np.max(np.abs(np.asarray(old[key])-np.asarray(new[key]))))
                       for key in ('energy', 'nuclear', 'source', 'rhs', 'zvector', 'rdm1', 'rdm2', 'X')})
    for key in paired[0]:
        assert max(p[key] for p in paired) < 1e-8, (key, paired)
    timings = {}
    for b in ('legacy', 'contracted'):
        ss = [s for s in samples if s['backend'] == b and s['repeat'] > 0]
        timings[b] = dict(total=float(np.median([s['seconds'] for s in ss])),
                         inner=float(np.median([s['inner_seconds'] for s in ss])),
                         source=float(np.median([s['stats']['seconds'] for s in ss])))
    result = dict(parameters=vars(args),
                  environment=dict(python=sys.version, pyscf=pyscf.__version__, numpy=np.__version__,
                                   platform=platform.platform(), threads=lib.num_threads()),
                  source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (ROOT/'embed_sim/dmet_nevpt/grad/gradients.py',
                                           ROOT/'embed_sim/grad/solvers.py',
                                           ROOT/'embed_sim/fic_nevpt/grad/ci_contraction.py')}, embedded_orbitals=int(dmet.nes), ci_size=mc.ci[0].size,
                  module_paths={m: importlib.import_module(m).__file__ for m in
                                ('embed_sim.ssdmet', 'embed_sim.grad.solvers',
                                 'embed_sim.dmet_nevpt.grad.gradients', 'embed_sim.sacasscf_mixer')},
                  max_differences={k: max(p[k] for p in paired) for k in paired[0]},
                  median_seconds=timings,
                  speedup={k: timings['legacy'][k]/timings['contracted'][k] for k in timings['legacy']},
                  samples=samples)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('max_differences', 'median_seconds', 'speedup')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependency-path')
    parser.add_argument('--optimizer', choices=('standard', 'newton'), default='standard')
    parser.add_argument('--spacing', type=float, default=1.1)
    parser.add_argument('--atoms', type=int, default=10)
    parser.add_argument('--impurity', type=int, default=4)
    parser.add_argument('--ncas', type=int, default=6)
    parser.add_argument('--canonstep', type=int, choices=(0, 1, 2), default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', default='tests/dmet_ci_benchmark.json')
    run(parser.parse_args())
