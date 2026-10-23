"""Main-library regression for fixed J/K caching, with the original solver."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from functools import partial
import unittest
from unittest.mock import patch
import numpy as np
from pyscf import gto, scf, lib
from embed_sim import aodmet, ssdmet
from embed_sim.grad import aodmet as aograd, ssdmet as ssgrad

lib.num_threads(1)
FACTORY = ssgrad.gen_contract_Z_A


class StaticResponseCacheTests(unittest.TestCase):
    def test_responses_and_jk_counts(self):
        for spin in (0, 2):
            for density_fit in (False, True):
                with self.subTest(spin=spin, density_fit=density_fit):
                    mol = gto.M(atom='H 0 0 0; H 0 0 1.4; H 0 4 .2; H .2 4 1.8',
                                basis='6-31g', unit='Bohr', spin=spin, verbose=0)
                    mf = scf.ROHF(mol)
                    if density_fit:
                        mf = mf.density_fit()
                    mf.run(conv_tol=1e-12)
                    self.assertTrue(mf.converged)
                    n = len(ssgrad.gen_get_indep_pair_slice(mf)(np.zeros((mol.nao, mol.nao))))
                    vectors = np.random.default_rng(31).normal(size=(3, n))
                    results = []
                    for cached in (False, True):
                        with patch.object(mf, 'get_jk', wraps=mf.get_jk) as jk:
                            contract = FACTORY(mf, cache_static=cached)
                            self.assertEqual(jk.call_count, 0)  # lazy initialization
                            results.append(np.array([contract(z) for z in vectors]))
                            fixed_calls = (1 if cached else 3) if spin else 0
                            self.assertEqual(jk.call_count, 3+fixed_calls)
                            # A new closure must initialize its own geometry cache.
                            fresh = FACTORY(mf, cache_static=cached)
                            fresh(vectors[0])
                            self.assertEqual(jk.call_count, 3+fixed_calls+1+bool(spin))
                    np.testing.assert_allclose(results[0], results[1], atol=1e-12, rtol=0)

    def test_full_gradients_and_finite_difference(self):
        xyz = np.array([[0.,0.,0.], [0.,0.,1.4], [0.,4.,.2], [.2,4.,1.8], [1.,2.,3.5]])
        for module, grad_module in ((aodmet, aograd), (ssdmet, ssgrad)):
            with self.subTest(module=module.__name__):
                def evaluate(cached, displacement=0.):
                    coords = xyz.copy(); coords[0, 2] += displacement
                    mol = gto.M(atom=[('H', r) for r in coords], basis='6-31g',
                                unit='Bohr', spin=1, verbose=0)
                    mf = scf.ROHF(mol).run(conv_tol=1e-12)
                    self.assertTrue(mf.converged)
                    cls = module.AODMET if module is aodmet else module.SSDMET
                    dmet = cls(mf, imp_idx=list(range(4)), threshold=1e-10, verbose=0)
                    dmet.build(save_chk=False)
                    grad = dmet.Gradients()
                    grad._krylov_conv_tol = 1e-10
                    grad.solver_option = dict(solver='CASSCF', ncas=dmet.es_mf.mo_coeff.shape[1],
                                              nelec=dmet.es_mf.mol.nelectron, fix_spin=False,
                                              conv_tol=1e-12, max_cycle=100)
                    with patch.object(grad_module, 'gen_contract_Z_A', partial(FACTORY, cache_static=cached)):
                        g, e = grad.kernel()
                    self.assertTrue(np.all(np.isfinite(g)))
                    return g, e
                old_g, old_e = evaluate(False)
                new_g, new_e = evaluate(True)
                np.testing.assert_allclose(new_g, old_g, atol=1e-10, rtol=0)
                self.assertLess(abs(new_e-old_e), 1e-12)
                h = .0005
                numeric = (evaluate(True, h)[1]-evaluate(True, -h)[1])/(2*h)
                self.assertLess(abs(numeric-new_g[0, 2]), 2e-6)

if __name__ == '__main__':
    unittest.main()
