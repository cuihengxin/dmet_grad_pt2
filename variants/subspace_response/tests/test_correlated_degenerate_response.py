"""Independent real-energy finite differences for a degenerate correlated bath."""
import unittest
import numpy as np
from check_degenerate_correlated_gradients import evaluate


class CorrelatedDegenerateResponseTests(unittest.TestCase):
    def test_lif_four_degenerate_values_small_external_gap(self):
        spec=dict(molecule='LiF',basis='sto-3g',bond=6.,imp_ao=0,fragments=2,threshold=1e-10)
        old=evaluate(spec,mode='legacy')
        new=evaluate(spec,mode='subspace')
        self.assertEqual(new['degenerate_count'],4)
        self.assertEqual(new['bath_count'],4)
        self.assertLess(new['external_gap'],1e-5)
        self.assertGreater(new['external_gap'],1e-6)
        self.assertEqual(new['ncas'],6)
        self.assertEqual(new['nelec'],8)
        self.assertLess(abs(old['energy']-new['energy']),1e-11)
        for h in (1e-3,5e-4):
            ep=evaluate(spec,h,energy_only=True)
            em=evaluate(spec,-h,energy_only=True)
            self.assertEqual(ep['bath_count'],new['bath_count'])
            self.assertEqual(em['bath_count'],new['bath_count'])
            self.assertEqual(ep['ncore'],new['ncore'])
            self.assertEqual(em['ncore'],new['ncore'])
            numeric=(ep['energy']-em['energy'])/(2*h)
            old_error=abs(numeric-old['gradient'][0][2])
            new_error=abs(numeric-new['gradient'][0][2])
            self.assertLess(new_error,2e-8)
            self.assertGreater(old_error,5e-5)
            self.assertLess(new_error,old_error/1000)
        np.testing.assert_allclose(np.sum(new['gradient'],axis=0),0.,atol=2e-7)

if __name__=='__main__':unittest.main()
