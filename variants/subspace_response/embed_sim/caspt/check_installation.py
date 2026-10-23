"""Smoke, relocation parity and finite-difference checks for caspt package."""
import argparse
import json
from pathlib import Path
import numpy as np
from pyscf import gto, scf, mcscf, lib
from caspt import CASPT2

def reference(distance, charge):
    atom = (f'N 0 0 0; H 0 {-0.82/lib.param.BOHR} {distance}; '
            f'H 0 {0.82/lib.param.BOHR} {0.65/lib.param.BOHR}') if charge else f'H 0 0 0; H 0 0 {distance}'
    mol = gto.M(atom=atom, unit='Bohr',
                basis='sto-3g', charge=0, spin=charge, verbose=0)
    mf = (scf.ROHF(mol) if charge else scf.RHF(mol)).run(conv_tol=1e-13)
    mc = mcscf.CASSCF(mf, 3 if charge else 2, (2, 1) if charge else (1, 1)).newton()
    mc.conv_tol, mc.conv_tol_grad = 1e-12, 1e-9
    mc.ah_conv_tol, mc.fcisolver.conv_tol = 1e-20, 1e-14
    mc.max_cycle_macro = 100
    mc.kernel()
    if not mc.converged:
        first = mc
        mc = mcscf.CASSCF(mf, 3 if charge else 2, (2, 1) if charge else (1, 1))
        mc.conv_tol, mc.conv_tol_grad = 1e-12, 1e-9
        mc.fcisolver.conv_tol = 1e-14
        mc.max_cycle_macro = 150
        mc.kernel(first.mo_coeff, first.ci)
    assert mc.converged
    return mc

def energy(mc, cls=CASPT2):
    pt = cls(mc)
    pt.conv_tol, pt.conv_tol_normt = 1e-12, 1e-10
    pt.max_cycle = 200
    pt.kernel()
    assert pt.converged
    return pt

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--compare-original', action='store_true')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    lib.num_threads(1)
    results = []
    for charge in (0, 1):
        print('Testing charge', charge, flush=True)
        distance = .65 / lib.param.BOHR if charge else .74 / lib.param.BOHR
        pt = energy(reference(distance, charge))
        grad = pt.nuc_grad_method().kernel()
        row = dict(case='nh2_doublet' if charge else 'h2_singlet', energy=pt.e_tot, gradient=grad.tolist())
        h = 0.0005
        ep, em = [energy(reference(distance+s*h, charge)).e_tot for s in (1, -1)]
        row['fd_error'] = abs((ep-em)/(2*h)-grad[1, 2])
        assert row['fd_error'] < 1e-6, row
        if args.compare_original:
            from pyscf.mrpt.caspt2 import CASPT2 as Original
            old = energy(pt._mc, Original)
            old_grad = old.nuc_grad_method().kernel()
            row['energy_delta'] = abs(old.e_tot-pt.e_tot)
            row['gradient_delta'] = float(np.max(np.abs(old_grad-grad)))
            assert row['energy_delta'] < 1e-12
            assert row['gradient_delta'] < 1e-12
        results.append(row)
    # Dense backend remains independently callable after relocation.
    mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g', verbose=0)
    mc = mcscf.CASSCF(scf.RHF(mol).run(), 2, 2).run()
    pt = energy(mc)
    delta = float(np.max(np.abs(pt.nuc_grad_method().kernel()
                  -pt.nuc_grad_method(backend='reference').kernel())))
    assert delta < 1e-8
    import pyscf
    out = dict(pyscf_path=pyscf.__file__,package_path=__import__('caspt').__file__,
               results=results,dense_backend_delta=delta)
    Path(args.output).write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
