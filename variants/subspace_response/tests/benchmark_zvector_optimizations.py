"""Factorial benchmark: static response caching and correction-equation solve."""
from benchmark_zvector import evaluate, ZVectorCache, aodmet, ssdmet
from benchmark_zvector_history import geometry
from embed_sim.grad.ssdmet import gen_contract_Z_A
from pyscf import gto, scf
import numpy as np
import json
import time
from pathlib import Path


def microbenchmark():
    results = []
    for df in (False, True):
        mol = gto.M(atom='O 0 0 0; O 0 0 2.3', unit='Bohr', basis='cc-pvdz', spin=2, verbose=0)
        mf = scf.ROHF(mol)
        if df:
            mf = mf.density_fit()
        mf.conv_tol = 1e-12
        mf.kernel()
        assert mf.converged
        occ = mf.mo_occ
        nd, nh, nv = sum(occ==2), sum((occ>0)&(occ<2)), sum(occ==0)
        rng = np.random.default_rng(41)
        vectors = rng.normal(size=(20, nh*nd+nv*nd+nv*nh))
        # Independent closures; include their setup and first uncached call.
        timings = {False: [], True: []}
        outputs = {}
        for repeat in range(5):
            for cached in ((False, True) if repeat % 2 == 0 else (True, False)):
                start = time.perf_counter()
                contract = gen_contract_Z_A(mf, cache_static=cached)
                outputs[cached] = np.array([contract(z) for z in vectors])
                timings[cached].append(time.perf_counter()-start)
                expected = 1 if cached else len(vectors)
                assert contract.stats['static_jk_calls'] == expected
        difference = float(np.max(np.abs(outputs[False]-outputs[True])))
        assert difference < 1e-10
        results.append(dict(system='O2 triplet/cc-pvdz', nao=mol.nao, density_fit=df,
                            actions=20, repeats=5, seconds={str(k):v for k,v in timings.items()},
                            max_response_difference=difference))
        print('MICRO', results[-1], flush=True)
    return results


def main():
    output = Path(__file__).with_name('zvector_optimization_results.json')
    import sys
    if '--micro-only' in sys.argv:
        result = json.loads(output.read_text())
        result['micro'] = microbenchmark()
        output.write_text(json.dumps(result, indent=2))
        return
    records, checks = [], []
    configs = [(False, False), (True, False), (False, True), (True, True)]
    for module in (aodmet, ssdmet):
        for spin in (0, 1):
            for repeat in range(5):
                ordered = configs[repeat % 4:]+configs[:repeat % 4]
                for cached, reuse in ordered:
                    cache = ZVectorCache()
                    for step in (0, 1):
                        result = evaluate(geometry(step, spin), module, 'krylov', True,
                                          cache, spin, 'previous', cached, reuse)
                        records.append(dict(module=module.__name__, spin=spin,
                            repeat=repeat, cache_static=cached, reuse_residual=reuse,
                            step=step, **result))
                    print(module.__name__, spin, repeat, cached, reuse,
                          result['zvector'], flush=True)
            rows = [r for r in records if r['module']==module.__name__ and r['spin']==spin]
            maxdiff = 0.
            for step in (0, 1):
                group = [r for r in rows if r['step']==step]
                ref = group[0]
                for r in group:
                    difference = float(np.max(np.abs(np.array(r['gradient'])-ref['gradient'])))
                    maxdiff = max(maxdiff, difference)
                    assert difference < 2e-8
                    assert abs(r['energy']-ref['energy']) < 1e-10
            ref = [r for r in rows if r['step']==1 and r['cache_static'] and r['reuse_residual']][0]
            fd = []
            for h in (.001, .0005):
                plus, minus = geometry(1, spin), geometry(1, spin)
                plus[0, 2] += h; minus[0, 2] -= h
                ep = evaluate(plus, module, 'krylov', False, ZVectorCache(), spin, cache_static=True, reuse_residual=True)['energy']
                em = evaluate(minus, module, 'krylov', False, ZVectorCache(), spin, cache_static=True, reuse_residual=True)['energy']
                fd.append(dict(step=h, error=abs((ep-em)/(2*h)-ref['gradient'][0][2])))
            assert fd[-1]['error'] < 2e-6
            checks.append(dict(module=module.__name__, spin=spin, max_gradient_difference=maxdiff, fd=fd))
            output.write_text(json.dumps(dict(threads=1, repeats=5, records=records, checks=checks), indent=2))
    micro = microbenchmark()
    output.write_text(json.dumps(dict(threads=1, repeats=5, records=records, checks=checks, micro=micro), indent=2))

if __name__ == '__main__':
    main()
