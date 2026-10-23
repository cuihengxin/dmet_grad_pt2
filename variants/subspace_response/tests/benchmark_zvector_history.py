"""Six-geometry Krylov sequence: zero, previous, minimum-residual history."""
from benchmark_zvector import evaluate, ZVectorCache, aodmet, ssdmet
import json
from pathlib import Path
import numpy as np


def geometry(step, spin):
    xyz = np.array([[0., 0., 0.], [0., 0., 1.4], [0., 4., .2], [.2, 4., 1.8]])
    if spin:
        xyz = np.vstack([xyz, [1., 2., 3.5]])
    xyz[0, 2] += .002*step
    xyz[2, 0] += .0002*step**2
    xyz[1, 1] += .0004*np.sin(.3*step)
    return xyz


def main():
    output = Path(__file__).with_name('zvector_history_results.json')
    records, checks = [], []
    for module in (aodmet, ssdmet):
        for spin in (0, 1):
            for repeat in range(3):
                modes = ['zero', 'previous', 'history']
                modes = modes[repeat:]+modes[:repeat]
                for mode in modes:
                    cache = ZVectorCache(max_history=3)
                    for step in range(6):
                        result = evaluate(geometry(step, spin), module, 'krylov',
                                          mode != 'zero', cache, spin,
                                          'history' if mode == 'history' else 'previous')
                        records.append(dict(module=module.__name__, spin=spin,
                            repeat=repeat, mode=mode, step=step, **result))
                    print(module.__name__, spin, repeat, mode,
                          records[-1]['zvector'], flush=True)
            rows = [r for r in records if r['module'] == module.__name__ and r['spin'] == spin]
            diff = 0.
            for step in range(6):
                group = [r for r in rows if r['step'] == step]
                ref = group[0]
                for r in group:
                    delta = float(np.max(np.abs(np.array(r['gradient'])-ref['gradient'])))
                    diff = max(diff, delta)
                    assert delta < 2e-8, (module.__name__, spin, step, r['mode'], delta)
                    assert abs(r['energy']-ref['energy']) < 1e-10
            last = [r for r in rows if r['step'] == 5 and r['mode'] == 'history'][0]
            fd = []
            for h in (.001, .0005):
                xyz = geometry(5, spin)
                plus, minus = xyz.copy(), xyz.copy()
                plus[0, 2] += h; minus[0, 2] -= h
                ep = evaluate(plus, module, 'krylov', False, ZVectorCache(), spin)['energy']
                em = evaluate(minus, module, 'krylov', False, ZVectorCache(), spin)['energy']
                fd.append(dict(step=h, error=abs((ep-em)/(2*h)-last['gradient'][0][2])))
            assert fd[-1]['error'] < 2e-6, fd
            checks.append(dict(module=module.__name__, spin=spin,
                               max_gradient_difference=diff, finite_differences=fd))
            output.write_text(json.dumps(dict(threads=1, basis='6-31g', history=3,
                geometry_steps=6, repeats=3, records=records, checks=checks), indent=2))
            print('CHECK', checks[-1], flush=True)

if __name__ == '__main__':
    main()
