"""Legacy/contracted CI-source equivalence, independent of CASPT2 modules."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
from pyscf import lib

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'embed_sim' if (root/'embed_sim').exists() else root))
from fic_nevpt.grad.gradients import Gradients
from fic_nevpt.grad.dfgradients import DFGradients
from embed_sim.dmet_nevpt.grad.gradients import Gradients as DMETGradients


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = lib.num_threads()
        lib.num_threads(1)

    @classmethod
    def tearDownClass(cls): lib.num_threads(cls.threads)

    def make_gradient(self, klass, norb, nelec, shape):
        rng = np.random.default_rng(100+norb+sum(nelec))
        ci = rng.normal(size=shape)
        ci /= np.linalg.norm(ci)
        grad = klass.__new__(klass)
        grad.max_memory = 2000
        grad.nevpt = SimpleNamespace(ncas=norb,nelecas=nelec,state_ci=ci,
                                     log=SimpleNamespace(debug=lambda *a:None))
        grad.rdms = SimpleNamespace(**{name:rng.normal(size=(norb,)*(2*rank))
            for rank,name in enumerate(('ca','ccaa','cccaaa','ccccaaaa'),1)})
        return grad

    def test_all_ranks_spin_and_both_gradient_classes(self):
        for klass in (Gradients,DFGradients,DMETGradients):
            for norb,nelec,shape in [(3,(2,2),(3,3)),(4,(2,1),(6,4)),
                                      (4,(3,1),(4,4)),(3,(1,0),(3,1))]:
                with self.subTest(klass=klass.__name__,nelec=nelec):
                    grad = self.make_gradient(klass,norb,nelec,shape)
                    grad.ci_response_backend = 'legacy'
                    expected = grad.get_ci_response()
                    self.assertEqual(grad.ci_response_stats['transition_rdm_calls'],np.prod(shape))
                    grad.ci_response_backend = 'contracted'
                    with mock.patch('fic_nevpt.grad.gradients.make_tran_rdm1234',
                                    side_effect=AssertionError('transition RDM call')):
                        with mock.patch('fic_nevpt.grad.dfgradients.make_tran_rdm1234',
                                        side_effect=AssertionError('transition RDM call')):
                            actual = grad.get_ci_response()
                    np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-12)
                    self.assertEqual(grad.ci_response_stats['transition_rdm_calls'],0)

    def test_dmet_contracted_avoids_transition_rdms(self):
        grad = self.make_gradient(DMETGradients, 3, (2, 2), (3, 3))
        with mock.patch('embed_sim.dmet_nevpt.grad.gradients.make_tran_rdm1234',
                        side_effect=AssertionError('transition RDM call')):
            grad.get_ci_response()
        self.assertEqual(grad.ci_response_stats['transition_rdm_calls'], 0)

    def test_invalid_backend(self):
        for klass in (Gradients, DMETGradients):
            grad = self.make_gradient(klass,3,(2,2),(3,3))
            grad.ci_response_backend = 'invalid'
            with self.assertRaises(ValueError): grad.get_ci_response()

    def test_memory_guard(self):
        for klass in (Gradients, DMETGradients):
            grad = self.make_gradient(klass,3,(2,2),(3,3))
            grad.max_memory = 0
            grad.ci_response_backend = 'contracted'
            with self.assertRaises(MemoryError): grad.get_ci_response()


if __name__=='__main__': unittest.main()
