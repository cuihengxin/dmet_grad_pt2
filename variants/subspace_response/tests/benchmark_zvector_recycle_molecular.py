"""Real O2/ROHF DF response; artificial smooth AO source, not an energy gradient."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import time
from types import SimpleNamespace
import numpy as np
from pyscf import gto, scf, lib
from embed_sim.grad._zvector import solve_zvector, ZVectorCache
from embed_sim.grad.ssdmet import gen_contract_Z_A, gen_get_indep_pair_slice
lib.num_threads(1)

class Log:
    def info(self, *args):
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subspace-tol", type=float, default=1e-16)
    args = parser.parse_args()
    rng = np.random.default_rng(51)
    records = []
    out = Path(__file__).with_name('zvector_recycle_molecular_results.json' if args.subspace_tol == 1e-16 else 'zvector_recycle_molecular_default_results.json')
    for distance in (2.3, 3.2, 4.0):
        models = []
        source = None
        for step in range(3):
            mol = gto.M(atom=[('O', (0.,0.,0.)), ('O', (0.,0.,distance+.001*step))],
                        basis='cc-pvdz', unit='Bohr', spin=2, verbose=0)
            mf = scf.ROHF(mol).density_fit().run(conv_tol=1e-12)
            assert mf.converged
            if source is None:
                source = rng.normal(size=(mol.nao,mol.nao)); source = source+source.T
            extract = gen_get_indep_pair_slice(mf)
            gaps = extract(mf.mo_energy[:,None]-mf.mo_energy[None,:])
            w = extract(mf.mo_coeff.T@source@mf.mo_coeff)
            models.append((mf, extract, gaps, w))
        for repeat in range(3):
            configs = [('zero',False,False), ('previous',True,False), ('recycle',True,True)]
            configs = configs[repeat:]+configs[:repeat]
            for name,warm,recycling in configs:
                cache = ZVectorCache()
                for step,(mf,extract,gaps,w) in enumerate(models):
                    grad = SimpleNamespace(base=SimpleNamespace(), zvector_cache=cache,
                        zvector_solver='krylov', zvector_warm_start=warm,
                        zvector_guess_mode='previous', zvector_reuse_residual=True,
                        zvector_recycle=recycling, zvector_recycle_dim=6,
                        zvector_recycle_min_iterations=12, zvector_restart=80,
                        zvector_max_cycle=200, _krylov_conv_tol=1e-7,
                        zvector_subspace_tol=args.subspace_tol, verbose=0)
                    t = time.perf_counter()
                    row = dict(distance_bohr=distance, step=step, repeat=repeat, configuration=name)
                    try:
                        z = solve_zvector(grad,mf,w,gaps,gen_contract_Z_A(mf),extract,Log())
                        row.update(success=True, **grad.zvector_stats)
                    except RuntimeError as exc:
                        row.update(success=False, error=str(exc), seconds=time.perf_counter()-t)
                    records.append(row)
                    print(row,flush=True)
        out.write_text(json.dumps(dict(system='O2 triplet/cc-pvdz density_fit ROHF',
            source='fixed random symmetric AO operator; not correlated-energy derivative',
            tolerance=1e-7, subspace_tolerance=args.subspace_tol, records=records),indent=2))

if __name__ == '__main__':
    main()
