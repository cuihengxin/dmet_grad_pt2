"""Projection and cross-geometry CAS warm-start regression tests."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from pyscf import gto, scf, mcscf, lib

spec = importlib.util.spec_from_file_location(
    'dmet_guess', Path(__file__).resolve().parents[1] / 'embed_sim/grad/guess.py')
guess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guess)


class GuessTests(unittest.TestCase):
    def test_nonorthogonal_embedding_metric(self):
        rng = np.random.default_rng(18)
        q, _ = np.linalg.qr(rng.normal(size=(7, 7)))
        s = np.diag(np.arange(1, 8))
        basis = np.diag(1 / np.sqrt(np.arange(1, 8))) @ q[:, :5]
        basis = basis @ np.diag([1., 2., 3., 4., 5.])
        orth_basis = np.diag(1 / np.sqrt(np.arange(1, 8))) @ q
        full = orth_basis[:, [5, 0, 1, 2, 3, 4, 6]]
        ref = np.diag([1., .5, 1/3, .25, .2])
        mo = guess.embedding_guess(basis, s, full, 2, 1, 2, ref)
        np.testing.assert_allclose(mo.T @ basis.T @ s @ basis @ mo,
                                   np.eye(5), atol=1e-12)
        # The active space is retained, even when an old frozen core is lost.
        np.testing.assert_allclose(basis @ mo[:, 1:3], full[:, 2:4], atol=1e-12)

    def test_active_rank_loss(self):
        with self.assertRaisesRegex(ValueError, 'active space loses rank'):
            guess.embedding_guess(np.eye(4)[:, :3], np.eye(4), np.eye(4),
                                  2, 1, 2, np.eye(3))

    def test_nevpt_reference_cache(self):
        options = dict(solver='NEVPT2', ncas=1, nelec=1)
        ref = SimpleNamespace(converged=True, mo_coeff=np.eye(2),
                              ci=[np.array([[.8], [.6]])], ncore=0,
                              ncas=1, nelecas=(1, 0))
        dmet = SimpleNamespace(fo_orb=np.empty((2, 0)), es_orb=np.eye(2),
                               fv_orb=np.empty((2, 0)), nfo=0,
                               mol=SimpleNamespace(copy=lambda: object()))
        # Deliberately different PT-level orbitals and CI must not be cached.
        pt = SimpleNamespace(mc=ref, mo_coeff=-np.eye(2), ci=None)
        cache = guess.CASGuessCache()
        self.assertFalse(cache.has_guess)
        cache.update(dmet, pt, options)
        self.assertTrue(cache.has_guess)
        np.testing.assert_array_equal(cache._snapshot['mo'], ref.mo_coeff)
        np.testing.assert_array_equal(cache._snapshot['ci'][0], ref.ci[0])
        ref.ci[0][:] = 0
        self.assertGreater(np.linalg.norm(cache._snapshot['ci'][0]), .9)
        cache.reset()
        self.assertFalse(cache.has_guess)

    def test_geometry_restart(self):
        threads = lib.num_threads()
        lib.num_threads(1)
        try:
            def make_dmet(distance):
                mol = gto.M(atom=[('H', (0, 0, i*distance)) for i in range(4)],
                            basis='sto-3g', verbose=0)
                mf = scf.RHF(mol).run()
                return SimpleNamespace(mol=mol, mf_or_cas=mf, es_mf=mf,
                                       es_orb=np.eye(4), fo_orb=np.empty((4, 0)),
                                       fv_orb=np.empty((4, 0)), nfo=0)
            options = dict(solver='CASSCF', ncas=2, nelec=2)
            old = make_dmet(1.2)
            mc = mcscf.CASSCF(old.es_mf, 2, 2).run()
            cache = guess.CASGuessCache()
            cache.update(old, mc, options)
            new = make_dmet(1.21)
            prepared = cache.prepare(new, options)
            np.testing.assert_allclose(prepared['init_guess'].T @ new.es_mf.get_ovlp()
                                       @ prepared['init_guess'], np.eye(4), atol=1e-10)
            restarted = mcscf.CASSCF(new.es_mf, 2, 2).run(
                prepared['init_guess'], ci0=prepared['ci0'])
            cold = mcscf.CASSCF(new.es_mf, 2, 2).run()
            self.assertTrue(restarted.converged)
            self.assertAlmostEqual(restarted.e_tot, cold.e_tot, places=8)
            np.testing.assert_array_equal(mc.ci, prepared['ci0'])
            prepared['ci0'][:] = 0
            self.assertGreater(np.linalg.norm(cache._snapshot['ci']), .9)
            snapshot = cache._snapshot
            restarted.converged = False
            cache.update(new, restarted, options)
            self.assertIs(cache._snapshot, snapshot)
            with self.assertRaisesRegex(ValueError, 'CAS settings changed'):
                cache.prepare(new, dict(options, ncas=3))
        finally:
            lib.num_threads(threads)


if __name__ == '__main__':
    unittest.main()
