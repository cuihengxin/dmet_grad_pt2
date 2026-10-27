"""Subspace-response regression using spectral and real correlated energy differences."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from pyscf import gto, scf, lib, mcscf
from embed_sim import aodmet, ssdmet
from embed_sim.grad import aodmet as aograd, ssdmet as ssgrad
from _embedding_response_legacy import legacy_response

lib.num_threads(1)
CASSCF = mcscf.CASSCF


def response(occ, nvir, nbath, damping=None, log=None):
    grad = SimpleNamespace(_perturb_anyway=damping is not None,
                           _perturbation=1e-10 if damping is None else damping)
    return ssgrad.get_embedding_response(grad, occ, nvir, nbath, log or Mock())


def tight_casscf(*args, **kwargs):
    mc = CASSCF(*args, **kwargs)
    mc.conv_tol_grad = 1e-9
    mc.fcisolver.conv_tol = 1e-14
    mc.fcisolver.max_cycle = 200
    mc.fcisolver.max_space = 50
    return mc


def molecular_evaluate(molecule, bond, displacement=0., legacy=False, energy_only=False, module=aodmet):
    # Two equal separated fragments give a reproducible degenerate bath.
    first, second = ('F', 'Li') if molecule == 'LiF' else ('H', 'H')
    xyz = np.array([[0., 1000.*i, z] for z in (0., bond) for i in range(2)])
    xyz[0, 2] += displacement
    mol = gto.M(atom=[(first if i < 2 else second, r) for i, r in enumerate(xyz)],
                basis='sto-3g' if molecule == 'LiF' else '6-31g', unit='Bohr', verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-10, max_cycle=200)
    if not mf.converged:
        raise RuntimeError('SCF not converged')
    # LiF: each F 1s; H2 control: the first two H atoms, all their AOs.
    impurity = [0, 5] if molecule == 'LiF' else list(range(4))
    cls = module.AODMET if module is aodmet else module.SSDMET
    dmet = cls(mf, imp_idx=impurity, threshold=1e-10, verbose=0)
    dmet.build(save_chk=False)
    ldm = dmet.lowdin_orth()[0]
    env = [i for i in range(mol.nao) if i not in dmet.imp_idx]
    occ = np.linalg.eigvalsh(ldm[np.ix_(env, env)])
    bath = occ[((occ >= 1e-10) & (occ <= 2-1e-10)) | (occ >= 2+1e-10)]
    ndegen = sum(len(g) for g in ssgrad.group_indexes_by_diff(bath, threshold=1e-8)[1])
    ncas = int(dmet.es_mf.mo_coeff.shape[1])
    nelec = int(dmet.es_mf.mol.nelectron)
    if energy_only:
        mc = tight_casscf(dmet.es_mf, ncas, nelec)
        mc.conv_tol = 1e-13
        mc.kernel()
        if not mc.converged:
            raise RuntimeError('FCI/CASSCF not converged')
        return float(mc.e_tot+dmet.fo_ene)
    grad = dmet.Gradients()
    grad._krylov_conv_tol = 1e-12
    grad.solver_option = dict(solver='CASSCF', ncas=ncas, nelec=nelec,
                              fix_spin=False, conv_tol=1e-13, max_cycle=100)
    grad_module = aograd if module is aodmet else ssgrad
    factory = legacy_response if legacy else ssgrad.get_embedding_response
    with patch.object(grad_module, 'get_embedding_response', factory), \
            patch.object(mcscf, 'CASSCF', tight_casscf):
        g, e = grad.kernel()
    if not grad.es_solver.converged:
        raise RuntimeError('FCI/CASSCF not converged')
    return g, float(e), ndegen


class EmbeddingResponseTests(unittest.TestCase):
    def test_internal_degeneracy_projector_and_gauge(self):
        rng = np.random.default_rng(731)
        q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        occ = np.array([0., 0., .4, .4, .4, .4, 2., 2.])
        v = rng.normal(size=(8, 8)); v = (v+v.T)/2
        a = rng.normal(size=(8, 8)); a = (a+a.T)/2
        factors, _ = response(occ, 2, 4)
        self.assertTrue(np.all(factors[:, 2:6] == 0))
        def derivative(vectors):
            db = vectors @ (factors*(vectors.T@v@vectors)[2:6]).T
            bath = vectors[:, 2:6]
            return db@bath.T+bath@db.T
        dp = derivative(q)
        h = (q*occ)@q.T
        for step in (1e-4, 5e-5):
            energies = []
            for sign in (1, -1):
                _, u = np.linalg.eigh(h+sign*step*v)
                energies.append(np.trace(a@u[:, 2:6]@u[:, 2:6].T))
            self.assertLess(abs((energies[0]-energies[1])/(2*step)-np.trace(a@dp)), 1e-5)
        r, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        rotated = q.copy(); rotated[:, 2:6] = q[:, 2:6]@r
        np.testing.assert_allclose(derivative(rotated), dp, atol=1e-12)
        p = q[:, 2:6]@q[:, 2:6].T
        np.testing.assert_allclose(p@dp+dp@p, dp, atol=1e-12)

    def test_external_gap_and_explicit_damping(self):
        log = Mock()
        factors, _ = response([0., .4, .4, 2.], 1, 2, log=log)
        log.warn.assert_not_called()
        np.testing.assert_allclose(factors[:, [0, 3]], [[2.5, -.625], [2.5, -.625]])
        factors, _ = response([0., 1e-6, 1.001e-6, .7, 2.], 1, 3)
        self.assertEqual(factors[0, 0], 1e6)
        self.assertTrue(np.all(factors[:, 1:4] == 0))
        damped, _ = response([0., 1e-6, 1.001e-6, .7, 2.], 1, 3, 1e-10)
        self.assertLess(damped[0, 0], 1e4)
        self.assertTrue(np.all(damped[:, 1:4] == 0))
        with self.assertRaises(ValueError):
            response([0., 0., 2.], 1, 1)
        factors, _ = response([0., 0., 2.], 1, 1, 1e-10)
        self.assertTrue(np.all(np.isfinite(factors)))
        log = Mock()
        response([0., 1e-9, 2.], 1, 1, log=log)
        log.warn.assert_called()

    def test_core_and_empty_spaces(self):
        _, core = response([0., .4, .4, 2.], 1, 2)
        np.testing.assert_allclose(core, [.5, .625, .625])
        factors, core = response([2.], 0, 1)  # no frozen core, no core response
        np.testing.assert_array_equal(factors, [[0.]])
        np.testing.assert_array_equal(core, [0.])
        factors, core = response([], 0, 0)
        self.assertEqual(factors.shape, (0, 0))
        self.assertEqual(core.shape, (0,))
        with self.assertRaises(ValueError):
            response([2., 2.], 0, 1)
        for damping in (-1., 0., np.nan):
            with self.assertRaises(ValueError):
                response([0., .4, 2.], 1, 1, damping)

    def test_correlated_lif_small_external_gap(self):
        for bond in (5., 6.):
            with self.subTest(bond=bond):
                old_g, old_e, _ = molecular_evaluate('LiF', bond, legacy=True)
                new_g, new_e, ndegen = molecular_evaluate('LiF', bond)
                self.assertEqual(ndegen, 4)
                self.assertLess(abs(new_e-old_e), 1e-11)
                for h in (1e-3, 5e-4):
                    ep = molecular_evaluate('LiF', bond, h, energy_only=True)
                    em = molecular_evaluate('LiF', bond, -h, energy_only=True)
                    numeric = (ep-em)/(2*h)
                    old_error, new_error = abs(numeric-old_g[0, 2]), abs(numeric-new_g[0, 2])
                    self.assertLess(new_error, 2e-8)
                    self.assertGreater(old_error, 2e-5)
                    self.assertLess(new_error, old_error/1000)
                np.testing.assert_allclose(new_g.sum(axis=0), 0., atol=2e-7)

    def test_degenerate_h2_control_both_implementations(self):
        for module in (aodmet, ssdmet):
            with self.subTest(module=module.__name__):
                old_g, old_e, _ = molecular_evaluate('H2', 1.4, legacy=True, module=module)
                new_g, new_e, ndegen = molecular_evaluate('H2', 1.4, module=module)
                self.assertEqual(ndegen, 2)
                np.testing.assert_allclose(new_g, old_g, atol=1e-8, rtol=0)
                self.assertLess(abs(new_e-old_e), 1e-11)
                h = 5e-4
                numeric = (molecular_evaluate('H2', 1.4, h, energy_only=True, module=module)-
                           molecular_evaluate('H2', 1.4, -h, energy_only=True, module=module))/(2*h)
                self.assertLess(abs(numeric-new_g[0, 2]), 1e-6)

if __name__ == '__main__':
    unittest.main()
