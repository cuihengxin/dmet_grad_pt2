"""Small real AODMET/NEVPT2 memory propagation check (H6/STO-3G).

Run: python tests/check_nevpt_memory.py
Optional: --dependency-path /path/to/complete/embed_sim
Calls the production embedded NEVPT solver and its real relaxed-RDM kernel.
Checks budget forwarding, not peak RSS or numerical gradient accuracy.
"""
import argparse
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from pyscf import gto, scf, lib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependency-path')
    parser.add_argument('--memory', type=float, nargs='+', default=[1536, 2304],
                        help='Algorithm budgets in MB; this does not allocate that memory.')
    parser.add_argument('--mode', choices=['initial', 'embedded'], default='initial',
                        help='initial: set only original mol/mf, then use scanner and build; '
                             'embedded: explicitly set ES-SCF as in the original test.')
    args = parser.parse_args()
    import embed_sim
    if args.dependency_path:
        embed_sim.__path__.append(str(Path(args.dependency_path).resolve()))
    from embed_sim import aodmet
    from embed_sim.grad import solvers
    from embed_sim.dmet_nevpt.grad.gradients import Gradients
    from embed_sim.fic_nevpt.grad import ci_contraction

    lib.num_threads(1)
    print('Production solver:', solvers.__file__, flush=True)
    original_kernel = Gradients.kernel
    original_contract = ci_contraction.contract_rdm_weights

    for budget in args.memory:
        assert budget > lib.current_memory()[0] + 100, 'Use a larger --memory budget'
        mol = gto.M(atom=[('H', (0, 0, i * 1.1 + 0.13 * (i % 2)))
                          for i in range(6)], basis='sto-3g', verbose=0)
        if args.mode == 'initial':
            mol.max_memory = budget
        mf = scf.ROHF(mol)
        mf.max_memory = mol.max_memory
        mf.kernel()
        assert mf.converged, 'Reference ROHF failed'
        scanner = mf.as_scanner()
        current = mol.copy()
        scanner(current)
        assert scanner.converged, 'Scanner ROHF failed'
        dmet = aodmet.AODMET(scanner, imp_idx=[0, 1, 2], threshold=1e-10, verbose=0)
        dmet.build(save_chk=False)
        if args.mode == 'embedded':
            # Deliberately different budgets test the explicit override alone.
            dmet.es_mf.max_memory = budget
            dmet.es_mf.mol.max_memory = 4000
        else:
            # NO assignments to dmet/es_mf/embedded mol memory in this mode.
            for name, value in [('mol', mol.max_memory), ('mf', mf.max_memory),
                                ('scanner', scanner.max_memory), ('DMET', dmet.max_mem),
                                ('ES-SCF', dmet.es_mf.max_memory)]:
                print('%s.max_memory = %g MB' % (name, value), flush=True)
                assert value == budget, '%s did not inherit budget' % name
        print('H6/STO-3G: AO=%d, embedding=%d, mode=%s'
              % (mol.nao, dmet.nes, args.mode), flush=True)
        seen = {}

        def checked_kernel(grad, *a, **kw):
            seen['gradient'] = float(grad.max_memory)
            seen['nevpt'] = float(grad.nevpt.max_memory)
            print('budget: ES-SCF=%g, ES-mol=%g, NEVPT=%g, gradient=%g MB'
                  % (dmet.es_mf.max_memory, dmet.es_mf.mol.max_memory,
                     grad.nevpt.max_memory, grad.max_memory), flush=True)
            assert grad.nevpt.max_memory == budget, 'NEVPT budget was not forwarded'
            assert grad.max_memory == budget, 'Gradient budget was not forwarded'
            return original_kernel(grad, *a, **kw)

        def checked_contract(*a, **kw):
            remaining = kw['max_memory']
            rss = lib.current_memory()[0]
            seen['remaining'] = remaining
            print('CI contraction: remaining=%.2f MB, current RSS=%.2f MB'
                  % (remaining, rss), flush=True)
            assert 0 < remaining <= budget, 'Invalid remaining CI budget'
            # RSS can change a little between production and wrapper samples.
            assert abs(remaining - max(0, budget - rss)) < 32, 'Budget subtraction mismatch'
            return original_contract(*a, **kw)

        options = dict(solver='NEVPT2', ncas=2, nelec=2, statelis=[2],
                       state=0, conv_tol=1e-10, max_cycle=100,
                       ci_response_backend='contracted')
        with patch.object(Gradients, 'kernel', checked_kernel), \
             patch.object(ci_contraction, 'contract_rdm_weights', checked_contract):
            solver, energy, dm1, dm2, x = solvers.NEVPT(dmet.es_mf, options)
        assert solver.mc.converged, 'CASSCF failed'
        assert 'remaining' in seen, 'Contracted CI response was not exercised'
        assert all(np.all(np.isfinite(v)) for v in (energy, dm1, dm2, x))
        print('PASS: %g MB reached NEVPT, gradient, and contracted CI; E=%.12f'
              % (budget, energy), flush=True)
    print('ALL MEMORY PROPAGATION CHECKS PASSED', flush=True)


if __name__ == '__main__':
    main()
