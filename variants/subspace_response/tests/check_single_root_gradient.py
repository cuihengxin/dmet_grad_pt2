"""Real AODMET single-root gradients and finite-difference regression.

python tests/check_single_root_gradient.py [--original-hessian]
--original-hessian restores the parent entry to reproduce list.ravel failure.
"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
from pyscf import gto, scf, lib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embed_sim import aodmet
from embed_sim.grad import aodmet as outer, solvers
from embed_sim.dmet_nevpt import NEVPT
from embed_sim.dmet_nevpt.grad.gradients import Gradients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-hessian', action='store_true')
    parser.add_argument('--output', default='tests/single_root_gradient_result.json')
    args = parser.parse_args()
    lib.num_threads(1)
    base_solve = Gradients.solve_lagrange
    solves = []

    def checked_solve(grad, *a, **kw):
        result = base_solve(grad, *a, **kw)
        assert result[0], 'NEVPT Z-vector failed to converge'
        solves.append(True)
        return result

    factory = solvers.sacasscf_mixer.sacasscf_mixer

    def tight_factory(*a, **kw):
        mc = factory(*a, **kw)
        mc.conv_tol_grad = 1e-6
        mc.max_cycle_macro = 300
        return mc

    solvers.sacasscf_mixer.sacasscf_mixer = tight_factory

    def evaluate(displacement=0., energy_only=False, spin=2, roots=None, backend='contracted'):
        coords = [(0., 0., 0.), (0., 0., 3.0)]
        atoms, basis, impurity, ncas, nelec = ['Li', 'H'], 'sto-3g', [0, 1, 2, 3], 3, 2
        if spin == 5:
            coords = [(0.05*(i % 2), 0., i*1.8+0.1*(i % 2)) for i in range(7)]
            atoms, basis, impurity, ncas, nelec = ['H']*7, '6-31g', list(range(8)), 6, 5
        coords[0] = (coords[0][0], coords[0][1], coords[0][2]+displacement)
        mol = gto.M(atom=list(zip(atoms, coords)), unit='Bohr',
                    basis=basis, spin=spin, verbose=0, max_memory=2000)
        mf = scf.ROHF(mol).run(conv_tol=1e-12)
        assert mf.converged
        dmet = aodmet.AODMET(mf, imp_idx=impurity, threshold=1e-10, verbose=0)
        dmet.build(save_chk=False)
        states = roots if roots is not None else [0, 0, 1]
        options = dict(solver='NEVPT2', ncas=ncas, nelec=nelec, statelis=states,
                       state=0, conv_tol=1e-12, max_cycle=100,
                       ci_response_backend=backend)
        if energy_only:
            mc = solvers.sacasscf_mixer.sacasscf_mixer(dmet.es_mf, ncas, nelec, states)
            mc.conv_tol = 1e-12
            mc.conv_tol_grad = 1e-6
            mc.max_cycle_macro = 300
            mc.kernel()
            assert mc.converged
            pt = NEVPT(mc, state=0)
            pt.kernel()
            return float(pt.e_tot+dmet.fo_ene)
        grad = dmet.Gradients()
        grad._krylov_conv_tol = 1e-10
        grad.solver_option = options
        with patch.object(Gradients, 'solve_lagrange', checked_solve):
            de, energy = grad.kernel()
        assert grad.es_solver.mc.converged
        assert np.all(np.isfinite(de))
        return float(energy), de

    if args.original_hessian:
        with patch.object(Gradients, 'get_Aop_Adiag',
                          super(Gradients, Gradients).get_Aop_Adiag):
            evaluate()
        raise AssertionError('Expected original single-root error was not reproduced')

    energy, gradient = evaluate()
    old_energy, old_gradient = evaluate(backend='legacy')
    np.testing.assert_allclose(gradient, old_gradient, atol=1e-8, rtol=0)
    assert abs(energy-old_energy) < 1e-10
    differences = []
    for h in [1e-3, 5e-4]:
        numeric = (evaluate(h, True)-evaluate(-h, True))/(2*h)
        error = abs(numeric-gradient[0, 2])
        print('FD', h, 'analytic', gradient[0, 2], 'numeric', numeric,
              'error', error, flush=True)
        assert error < 2e-5, 'Finite-difference gradient mismatch'
        differences.append(dict(step_bohr=h, analytic=float(gradient[0, 2]),
                                numerical=numeric, error=error))
    # Two-root path must remain identical to the inherited implementation.
    e_multi, g_multi = evaluate(spin=0, roots=[2])
    with patch.object(Gradients, 'get_Aop_Adiag',
                      super(Gradients, Gradients).get_Aop_Adiag):
        e_parent, g_parent = evaluate(spin=0, roots=[2])
    np.testing.assert_allclose(g_multi, g_parent, atol=1e-10, rtol=0)
    assert abs(e_multi-e_parent) < 1e-12
    sextet_states = [0, 0, 0, 0, 0, 1]
    e_six, g_six = evaluate(spin=5, roots=sextet_states)
    h = 5e-4
    numeric_six = (evaluate(h, True, spin=5, roots=sextet_states)
                   - evaluate(-h, True, spin=5, roots=sextet_states))/(2*h)
    error_six = abs(numeric_six-g_six[0, 2])
    assert error_six < 2e-5, 'Sextet finite-difference mismatch'
    result = dict(single_root_energy=energy, finite_differences=differences,
                  legacy_contracted_gradient_max_difference=float(np.max(np.abs(gradient-old_gradient))),
                  multiroot_gradient_max_difference=float(np.max(np.abs(g_multi-g_parent))),
                  sextet=dict(energy=e_six, analytic=float(g_six[0, 2]),
                              numerical=numeric_six, error=error_six),
                  converged_response_solves=len(solves))
    result['python'] = sys.version
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print('SINGLE ROOT GRADIENT CHECKS PASSED', flush=True)


if __name__ == '__main__':
    main()
