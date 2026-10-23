"""Independent finite-difference and integration checks for embedding response.

Run: python -m unittest discover -s tests -p test_embedding_response.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from pyscf import lib
from embed_sim.grad._embedding_response import embedding_response_factors, response_factors


class EmbeddingResponseTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(731)
        self.q = np.linalg.qr(rng.normal(size=(8, 8)))[0]
        a = rng.normal(size=(8, 8))
        self.v = (a + a.T) / 2
        a = rng.normal(size=(8, 8))
        self.observable = (a + a.T) / 2

    def check_spectral_energy(self, occ):
        """Differentiate Tr(A P_bath), using only eigh for reference energies."""
        occ = np.asarray(occ)
        q, v, a = self.q, self.v, self.observable
        h = q @ np.diag(occ) @ q.T
        factors, _ = embedding_response_factors(occ, 2, 4)
        db = q @ (factors * (q.T @ v @ q)[2:6]).T
        dp = db @ q[:, 2:6].T + q[:, 2:6] @ db.T
        analytic = np.trace(a @ dp)
        errors = []
        for step in (1e-4, 5e-5):
            energies = []
            for sign in (1, -1):
                _, u = np.linalg.eigh(h + sign * step * v)
                energies.append(np.trace(a @ u[:, 2:6] @ u[:, 2:6].T))
            errors.append(abs((energies[0]-energies[1])/(2*step) - analytic))
        self.assertLess(errors[-1], 2e-6)
        self.assertLess(errors[-1], errors[0] * .4 + 1e-9)
        # Tangent of P^2=P: an independent projector consistency condition.
        p = q[:, 2:6] @ q[:, 2:6].T
        np.testing.assert_allclose(p @ dp + dp @ p, dp, atol=1e-12)

    def test_exact_fourfold_degenerate_energy_derivative(self):
        self.check_spectral_energy([0., 0., .4, .4, .4, .4, 2., 2.])

    def test_near_degenerate_energy_derivative(self):
        self.check_spectral_energy([0., 0., .4, .4+1e-10, .4+2e-10, .9, 2., 2.])

    def test_nondegenerate_energy_derivative(self):
        self.check_spectral_energy([0., 0., .2, .4, .7, 1.2, 2., 2.])

    def test_small_external_gap_against_finite_difference(self):
        # Internal near degeneracy triggers legacy's global damping. Check the
        # resulting projector against independently recomputed eigenvectors.
        from check_embedding_response import legacy
        occ = np.array([0., 0., 1e-6, 1.001e-6, .7, .9, 2., 2.])
        q = self.q
        h = q @ np.diag(occ) @ q.T
        step = 1e-11
        projections = []
        for sign in (1, -1):
            _, u = np.linalg.eigh(h + sign*step*self.v)
            projections.append(u[:, 2:6] @ u[:, 2:6].T)
        numerical = (projections[0] - projections[1]) / (2*step)
        errors = []
        for factors in (embedding_response_factors(occ, 2, 4)[0], legacy(occ, 2, 4)[0]):
            db = q @ (factors * (q.T @ self.v @ q)[2:6]).T
            dp = db @ q[:, 2:6].T + q[:, 2:6] @ db.T
            errors.append(np.linalg.norm(dp - numerical) / np.linalg.norm(numerical))
        self.assertLess(errors[0], 1e-4)
        self.assertGreater(errors[1], .1)

    def test_invalid_inputs(self):
        for occ, nv, nb, damping in [([np.nan], 0, 1, None),
                                      ([[.4]], 0, 1, None),
                                      ([.4], -1, 1, None),
                                      ([.4], 0, 2, None),
                                      ([.4], 0, 1, -1.),
                                      ([.4], 0, 1, np.inf)]:
            with self.subTest(occupations=occ, nvir=nv, nbath=nb, damping=damping):
                with self.assertRaises(ValueError):
                    embedding_response_factors(occ, nv, nb, damping)

    def test_core_projector_derivative(self):
        occ = np.array([0., 0., .4, .4, .4, .4, 2., 2.])
        q = self.q
        h = q @ np.diag(occ) @ q.T
        _, factors = embedding_response_factors(occ, 2, 4)
        dc = q[:, :6] @ (factors[:, None] * (q.T @ self.v @ q)[:6, 6:])
        dp = dc @ q[:, 6:].T + q[:, 6:] @ dc.T
        step = 1e-5
        projections = []
        for sign in (1, -1):
            _, u = np.linalg.eigh(h + sign*step*self.v)
            projections.append(u[:, 6:] @ u[:, 6:].T)
        np.testing.assert_allclose((projections[0]-projections[1])/(2*step), dp,
                                   atol=1e-8, rtol=1e-7)

    def test_external_degeneracy_requires_explicit_damping(self):
        with self.assertRaisesRegex(ValueError, 'not unique'):
            embedding_response_factors([0., 0., 2.], 1, 1)
        w, c = embedding_response_factors([0., 0., 2.], 1, 1, 1e-10)
        self.assertTrue(np.isfinite(w).all() and np.isfinite(c).all())
        self.assertEqual(w[0, 0], 0.)

    def test_empty_subspaces(self):
        for occ, nv, nb in [([], 0, 0), ([0., 2.], 1, 0), ([.4, .4], 0, 2), ([2.], 0, 1)]:
            w, c = embedding_response_factors(occ, nv, nb)
            self.assertEqual(w.shape, (nb, len(occ)))
            self.assertTrue(np.isfinite(w).all() and np.isfinite(c).all())
        np.testing.assert_array_equal(embedding_response_factors([2.], 0, 1)[1], [0.])

    def test_dispatch_and_warnings(self):
        grad = SimpleNamespace(embedding_response='subspace', _perturb_anyway=False,
                               _perturbation=1e-10)
        log = Mock()
        for implementation in ('aodmet', 'ssdmet'):
            response_factors(grad, np.array([0., .4, .4, 2.]), 1, 2, log, implementation)
            log.warn.assert_not_called()  # Internal degeneracy alone is harmless.
            grad.embedding_response = 'legacy'
            old = response_factors(grad, np.array([0., .4, .8, 2.]), 1, 2, log, implementation)
            grad.embedding_response = 'subspace'
            new = response_factors(grad, np.array([0., .4, .8, 2.]), 1, 2, log, implementation)
            np.testing.assert_allclose(old[0][:, [0, 3]], new[0][:, [0, 3]])
            np.testing.assert_allclose(old[1], new[1])
        response_factors(grad, np.array([0., 1e-9, 2.]), 1, 1, log, 'aodmet')
        log.warn.assert_called()
        grad.embedding_response = 'invalid'
        with self.assertRaises(ValueError):
            response_factors(grad, np.array([0., .4, 2.]), 1, 1, log, 'aodmet')

    def test_spectral_gauge_and_damping_regression(self):
        from check_embedding_response import spectral_checks
        spectral_checks()

    def test_fourfold_degenerate_molecular_gradients(self):
        from check_embedding_response import molecular_checks
        threads = lib.num_threads()
        lib.num_threads(1)
        try:
            molecular_checks()
        finally:
            lib.num_threads(threads)


if __name__ == '__main__':
    unittest.main()
