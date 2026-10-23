"""Controlled many-iteration response problem: cold, previous Z, recycled space."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from types import SimpleNamespace
import numpy as np
from pyscf import gto
from embed_sim.grad._zvector import ZVectorCache, solve_zvector, _blocks
from embed_sim.grad.ssdmet import gen_get_indep_pair_slice


class Log:
    def info(self, *args):
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=8)
    args = parser.parse_args()
    mol = gto.M(atom=[('H', (0., 0., 2*i)) for i in range(14)], unit='Bohr', basis='6-31g', verbose=0)
    e, u = np.linalg.eigh(mol.intor('int1e_ovlp'))
    mf = SimpleNamespace(mol=mol, mo_coeff=(u/e**.5)@u.T,
                         mo_occ=np.r_[np.full(7, 2.), np.zeros(21)])
    extract = gen_get_indep_pair_slice(mf)
    gaps = extract(np.arange(28)[:, None]-np.arange(28)[None, :]).astype(float)
    n = len(gaps)
    rng = np.random.default_rng(2026)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigenvalues = np.r_[np.geomspace(.03, .15, 8), np.linspace(.4, 1.8, n-8)]
    a = (q*eigenvalues)@q.T
    b0 = rng.normal(size=n)*10
    direction = rng.normal(size=n)
    records = []
    for repeat in range(3):
        configs = [('zero', False, False), ('previous', True, False), ('recycle', True, True)]
        configs = configs[repeat:]+configs[:repeat]
        for name, warm, recycle in configs:
            cache = ZVectorCache()
            for step in range(3):
                # Fixed response matrix isolates changed RHS; changing-matrix
                # case is exercised separately below using the same spectrum.
                current_a = a if step == 0 else a+step*1e-5*np.eye(n)
                b = b0+.1*step*direction
                w = -gaps*b
                def contract(z):
                    v = gaps*((current_a-np.eye(n))@z)
                    out, offset = np.zeros((28, 28)), 0
                    for left, right in _blocks(mf.mo_occ):
                        size = len(left)*len(right)
                        out[np.ix_(left, right)] = v[offset:offset+size].reshape(len(left), len(right))
                        offset += size
                    return out
                grad = SimpleNamespace(base=SimpleNamespace(), zvector_cache=cache,
                    zvector_solver='krylov', zvector_warm_start=warm,
                    zvector_recycle=recycle, zvector_recycle_dim=args.dim,
                    zvector_recycle_min_iterations=12, zvector_reuse_residual=True,
                    zvector_guess_mode='previous', zvector_restart=80,
                    zvector_max_cycle=200, _krylov_conv_tol=1e-7, zvector_subspace_tol=1e-16, verbose=0)
                z = solve_zvector(grad, mf, w, gaps, contract, extract, Log())
                dense = np.linalg.solve(current_a, b)
                error = float(np.linalg.norm(z-dense)/np.linalg.norm(dense))
                assert error < 1e-8, error
                records.append(dict(repeat=repeat, configuration=name, step=step,
                                    relative_solution_error=error, **grad.zvector_stats))
                print(name, step, grad.zvector_stats, 'relative_error', error, flush=True)
    Path(__file__).with_name(f'zvector_recycle_synthetic_dim{args.dim}_results.json').write_text(
        json.dumps(dict(dimension=n, eigenvalue_min=float(eigenvalues.min()),
                        eigenvalue_max=float(eigenvalues.max()), records=records), indent=2))

if __name__ == '__main__':
    main()
