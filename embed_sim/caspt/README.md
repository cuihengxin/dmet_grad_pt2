# IC-CASPT2 energy and analytic gradients

This package copies the implementation from the sibling PySCF development
tree, retaining its original license notices. Internal CASPT imports are
relative: installing custom modules into `pyscf.mrpt` or `pyscf.grad` is not
required. NumPy, SciPy, h5py and a working PySCF installation are required.
The original implementation remains unchanged. This is a copy, not a symlink;
future changes must be synchronized deliberately.

Add `/home/cuihx/Desktop/17grad/nevpt_grad/embed_sim` to PYTHONPATH, then:

```python
from pyscf import gto, scf, mcscf
from caspt import CASPT2

mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='sto-3g')
mf = scf.RHF(mol).run()
mc = mcscf.CASSCF(mf, 2, 2).run()
pt = CASPT2(mc)
pt.kernel()
gradient = pt.nuc_grad_method().kernel()

# Existing small-system dense implementation remains available:
# gradient = pt.nuc_grad_method(backend='reference').kernel()

# Optional geometry optimization (requires geomeTRIC):
# from pyscf.geomopt.geometric_solver import optimize
# optimized_mol = optimize(pt.nuc_grad_method())
```

If `nevpt_grad` itself is importable, `from embed_sim.caspt import CASPT2`
also works; do not mix both import styles in one calculation.

Gradient limitations: real single-state, converged all-electron CASSCF,
no frozen orbitals, no DF, no ECP, no symmetry, IPEA=0. The default gradient
is matrix-free for amplitudes and response, but retains conventional full
integrals and explicit active 4-RDMs. The dense reference backend is intended
only for tiny validation cases. The existing open-shell NH Z-vector failure
and CH2 coarse finite-difference anomaly are not fixed by this relocation.
No large-system performance claim is implied.
