import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from types import SimpleNamespace
import numpy as np
from pyscf import gto, scf
from embed_sim.grad._zvector import ZVectorCache, solve_zvector, _blocks
from embed_sim.grad.ssdmet import gen_get_indep_pair_slice


class Log:
    def info(self, *args):
        pass


class ZVectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='H 0 0 0; H 0 0 1.4; H 0 4 0; H 0 4 1.4',
                        basis='6-31g', unit='Bohr', verbose=0)
        s = cls.mol.intor('int1e_ovlp')
        e, u = np.linalg.eigh(s)
        cls.c = (u/e**.5)@u.T

    def mf(self):
        return SimpleNamespace(mol=self.mol, mo_coeff=self.c.copy(),
                               mo_occ=np.array([2., 2., 1., 0., 0., 0., 0., 0.]))

    def problem(self, method='gmres', warm=False):
        mf = self.mf()
        extract = gen_get_indep_pair_slice(mf)
        gaps = extract(np.arange(8)[:, None]-np.arange(8)[None, :])+.3
        rng = np.random.default_rng(14)
        k = rng.normal(size=(len(gaps), len(gaps)))*.035
        w = rng.normal(size=len(gaps))
        def contract(z):
            result = np.zeros((8, 8))
            offset = 0
            v = gaps*(k@z)
            for left, right in _blocks(mf.mo_occ):
                size = len(left)*len(right)
                result[np.ix_(left, right)] = v[offset:offset+size].reshape(len(left), len(right))
                offset += size
            return result
        grad = SimpleNamespace(base=SimpleNamespace(), zvector_solver=method,
            zvector_warm_start=warm, zvector_restart=30, zvector_max_cycle=200,
            _krylov_conv_tol=1e-10, verbose=0)
        return mf, extract, gaps, k, w, contract, grad

    def test_gmres_and_krylov_match_dense(self):
        for method in ('gmres', 'krylov'):
            mf, extract, gaps, k, w, contract, grad = self.problem(method)
            if method == 'krylov':
                grad._krylov_conv_tol = 1e-8
            z = solve_zvector(grad, mf, w, gaps, contract, extract, Log())
            np.testing.assert_allclose(z, np.linalg.solve(np.eye(len(w))+k, -w/gaps), atol=1e-7)
            self.assertLess(grad.zvector_stats['scaled_residual'], 1e-7)

    def test_transport_rotation_and_sign(self):
        mf = self.mf()
        rng = np.random.default_rng(28)
        rotations = []
        u = np.zeros((8, 8))
        for group in (np.arange(2), np.arange(2, 3), np.arange(3, 8)):
            q, _ = np.linalg.qr(rng.normal(size=(len(group), len(group))))
            u[np.ix_(group, group)] = q
        z = rng.normal(size=17)
        cache = ZVectorCache(); cache.update(mf, z)
        mf.mo_coeff = mf.mo_coeff@u
        guess, reason = cache.guess(mf)
        expected, offset = [], 0
        for left, right in _blocks(mf.mo_occ):
            size = len(left)*len(right)
            block = z[offset:offset+size].reshape(len(left), len(right))
            expected.extend((u[np.ix_(left, left)].T@block@u[np.ix_(right, right)]).ravel())
            offset += size
        np.testing.assert_allclose(guess, expected, atol=1e-13)
        self.assertEqual(reason, 'transported')
        mf.mo_occ[2] = 0
        self.assertIsNone(cache.guess(mf)[0])

    def test_warm_start_reduces_work_and_bad_guess_rejected(self):
        mf, extract, gaps, k, w, contract, grad = self.problem(warm=True)
        z = solve_zvector(grad, mf, w, gaps, contract, extract, Log())
        cold = grad.zvector_stats['matvecs']
        z2 = solve_zvector(grad, mf, w, gaps, contract, extract, Log())
        np.testing.assert_allclose(z2, z, atol=1e-10)
        self.assertLess(grad.zvector_stats['matvecs'], cold)
        grad.base._zvector_cache.update(mf, z*1e9)
        solve_zvector(grad, mf, w, gaps, contract, extract, Log())
        self.assertTrue(grad.zvector_stats['warm_start'].startswith('rejected'))

    def test_krylov_warm_start(self):
        mf, extract, gaps, k, w, contract, grad = self.problem('krylov', True)
        z = solve_zvector(grad, mf, w, gaps, contract, extract, Log())
        cold = grad.zvector_stats.copy()
        # Nearby right-hand side models a small geometry change; the molecular
        # benchmark independently exercises actual cross-geometry transport.
        z2 = solve_zvector(grad, mf, w*1.00001, gaps, contract, extract, Log())
        np.testing.assert_allclose(z2, np.linalg.solve(np.eye(len(w))+k, -w*1.00001/gaps), atol=1e-8)
        self.assertEqual(grad.zvector_stats['warm_start'], 'transported')
        self.assertLess(grad.zvector_stats['iterations'], cold['iterations'])
        self.assertLess(grad.zvector_stats['matvecs'], cold['matvecs'])
        self.assertLess(grad.zvector_stats['scaled_residual'], 1e-9)

    def test_history_combination_matches_dense(self):
        mf, extract, gaps, k, w, contract, grad = self.problem('krylov', True)
        cache = ZVectorCache(max_history=3)
        rng = np.random.default_rng(84)
        w2 = rng.normal(size=len(w))
        a = np.eye(len(w))+k
        cache.update(mf, np.linalg.solve(a, -w/gaps))
        cache.update(mf, np.linalg.solve(a, -w2/gaps))
        grad.zvector_cache = cache
        grad.zvector_guess_mode = 'history'
        target = 2*w-.5*w2
        z = solve_zvector(grad, mf, target, gaps, contract, extract, Log())
        np.testing.assert_allclose(z, np.linalg.solve(a, -target/gaps), atol=1e-12)
        self.assertEqual(grad.zvector_stats['history_count'], 2)
        self.assertEqual(grad.zvector_stats['history_rank'], 2)
        self.assertEqual(grad.zvector_stats['iterations'], 0)
        self.assertLess(grad.zvector_stats['initial_scaled_residual'], 1e-12)
        self.assertEqual(grad.zvector_stats['warm_start'], 'history: least squares')

    def test_history_dependent_vectors_and_reset(self):
        mf, extract, gaps, k, w, contract, grad = self.problem('krylov', True)
        cache = ZVectorCache(max_history=3)
        z = np.linalg.solve(np.eye(len(w))+k, -w/gaps)
        for _ in range(5):
            cache.update(mf, z)
        self.assertEqual(len(cache.history), 3)
        grad.zvector_cache = cache
        grad.zvector_guess_mode = 'history'
        result = solve_zvector(grad, mf, w*1.2, gaps, contract, extract, Log())
        np.testing.assert_allclose(result, 1.2*z, atol=1e-12)
        self.assertEqual(grad.zvector_stats['history_rank'], 1)
        mf.mo_occ[2] = 0
        cache.update(mf, z)
        self.assertEqual(len(cache.history), 1)
        cache.reset()
        self.assertEqual(len(cache.history), 0)
        self.assertIsNone(cache.data)
        with self.assertRaises(ValueError):
            ZVectorCache(max_history=0)

    def test_residual_reuse_saves_one_action(self):
        for method in ('krylov', 'gmres'):
            for mode in ('previous', 'history'):
                mf, extract, gaps, k, w, contract, grad = self.problem(method, True)
                a = np.eye(len(w))+k
                target = w+np.linspace(-1e-4, 1e-4, len(w))
                results, stats = [], []
                for reuse in (False, True):
                    cache = ZVectorCache()
                    cache.update(mf, np.linalg.solve(a, -w/gaps))
                    grad.zvector_cache = cache
                    grad.zvector_guess_mode = mode
                    grad.zvector_reuse_residual = reuse
                    results.append(solve_zvector(grad, mf, target, gaps, contract, extract, Log()))
                    stats.append(grad.zvector_stats.copy())
                np.testing.assert_allclose(results[0], results[1], atol=1e-12)
                np.testing.assert_allclose(results[1], np.linalg.solve(a, -target/gaps), atol=1e-9)
                self.assertEqual(stats[0]['matvecs']-stats[1]['matvecs'], 1)
                self.assertEqual(stats[0]['iterations'], stats[1]['iterations'])
                self.assertTrue(stats[1]['residual_reused'])

    def test_static_jk_cache_same_response(self):
        from embed_sim.grad.ssdmet import gen_contract_Z_A
        from unittest.mock import patch
        for spin in (0, 2):
            mol = self.mol.copy()
            mol.spin = spin
            mf = scf.ROHF(mol).run(conv_tol=1e-12)
            self.assertTrue(mf.converged)
            extract = gen_get_indep_pair_slice(mf)
            n = len(extract(np.zeros((mol.nao, mol.nao))))
            rng = np.random.default_rng(31)
            vectors = rng.normal(size=(3, n))
            values = []
            for cached in (False, True):
                with patch.object(mf, 'get_jk', wraps=mf.get_jk) as jk:
                    contract = gen_contract_Z_A(mf, cache_static=cached)
                    values.append(np.array([contract(z) for z in vectors]))
                    fixed_calls = (1 if cached else 3) if spin else 0
                    self.assertEqual(contract.stats['static_jk_calls'], fixed_calls)
                    self.assertEqual(jk.call_count, 3+fixed_calls)
            np.testing.assert_allclose(values[0], values[1], atol=1e-12)

    def test_recycled_deflated_equation_matches_dense(self):
        from embed_sim.grad._zvector_recycle import slow_directions, response_basis
        v = np.eye(5)
        a = np.diag([.1, .2, .7, 1., 1.4])
        u = slow_directions(list(v.T), list((a@v).T), 2)
        np.testing.assert_allclose(u@u.T, np.diag([1., 1., 0., 0., 0.]), atol=1e-12)
        b, c = response_basis(u, a@u)
        np.testing.assert_allclose(a@b, c, atol=1e-12)
        np.testing.assert_allclose(c.T@c, np.eye(2), atol=1e-12)
        mf, extract, gaps, k, w, contract, grad = self.problem('krylov', True)
        cache = ZVectorCache()
        cache.update_recycle(mf, np.eye(len(w))[:, :3])
        grad.zvector_cache = cache
        grad.zvector_recycle = True
        grad.zvector_recycle_dim = 3
        grad.zvector_recycle_min_iterations = 1
        z = solve_zvector(grad, mf, w, gaps, contract, extract, Log())
        np.testing.assert_allclose(z, np.linalg.solve(np.eye(len(w))+k, -w/gaps), atol=1e-8)
        self.assertTrue(grad.zvector_stats['recycle_active'])
        self.assertEqual(grad.zvector_stats['recycle_setup_matvecs'], 3)
        self.assertEqual(grad.zvector_stats['recycle_stored_dim'], 3)
        # Matrix transport must handle multiple directions at once.
        transported = cache.recycled(mf)
        np.testing.assert_allclose(transported.T@transported, np.eye(3), atol=1e-10)
        cache.reset()
        self.assertIsNone(cache.recycled(mf))

    def test_recycled_real_molecular_gradient(self):
        from benchmark_zvector import evaluate, aodmet, ssdmet
        xyz = np.array([[0.,0.,0.], [0.,0.,1.4], [0.,4.,.2], [.2,4.,1.8], [1.,2.,3.5]])
        for module in (aodmet, ssdmet):
            cache = ZVectorCache()
            settings = dict(spin=1, cache_static=True, reuse_residual=True,
                            recycle=True, recycle_dim=2, recycle_min_iterations=1)
            first = evaluate(xyz, module, 'krylov', True, cache, **settings)
            self.assertEqual(first['zvector']['recycle_stored_dim'], 2)
            nxt = xyz.copy(); nxt[0, 2] += .001
            second = evaluate(nxt, module, 'krylov', True, cache, **settings)
            baseline = evaluate(nxt, module, 'krylov', False, ZVectorCache(), spin=1,
                                cache_static=True, reuse_residual=True)
            self.assertTrue(second['zvector']['recycle_active'])
            np.testing.assert_allclose(second['gradient'], baseline['gradient'], atol=2e-8, rtol=0)
            self.assertLess(abs(second['energy']-baseline['energy']), 1e-10)

    def test_nonconvergence_raises(self):
        mf, extract, gaps, k, w, contract, grad = self.problem()
        grad.zvector_max_cycle = 1
        with self.assertRaises(RuntimeError):
            solve_zvector(grad, mf, w, gaps, contract, extract, Log())
        gaps[0] = 0
        with self.assertRaises(ValueError):
            solve_zvector(grad, mf, w, gaps, contract, extract, Log())

if __name__ == '__main__':
    unittest.main()
