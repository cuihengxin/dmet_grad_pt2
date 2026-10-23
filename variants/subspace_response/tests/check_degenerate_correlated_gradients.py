"""Real correlated molecular gradients with measured degenerate bath spectra."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import time
from unittest.mock import patch
import numpy as np
from pyscf import gto,scf,lib
from embed_sim import aodmet
from embed_sim.grad import aodmet as outer, solvers
lib.num_threads(1)
BASE_CASSCF=solvers.mcscf.CASSCF


def tight_casscf(*args, **kwargs):
    mc=BASE_CASSCF(*args, **kwargs)
    mc.conv_tol_grad=1e-9
    mc.fcisolver.conv_tol=1e-14
    mc.fcisolver.max_cycle=200
    mc.fcisolver.max_space=50
    return mc


def evaluate(spec, delta=0., mode='subspace', energy_only=False, axis=2, legacy_damping=None):
    n=spec.get('fragments',4); separation=1000.
    first,second=('H','H') if spec['molecule']=='H2' else ('F','Li')
    xyz=np.array([[0.,separation*i,z] for z in (0.,spec['bond']) for i in range(n)])
    xyz[0,axis]+=delta
    atoms=[(first if i<n else second,r) for i,r in enumerate(xyz)]
    mol=gto.M(atom=atoms,basis=spec['basis'],unit='Bohr',verbose=0)
    mf=scf.ROHF(mol).run(conv_tol=1e-14,conv_tol_grad=1e-10,max_cycle=200)
    if not mf.converged:
        mf=mf.newton().run(conv_tol=1e-14,conv_tol_grad=1e-10,max_cycle=100)
    if not mf.converged: raise RuntimeError('SCF not converged')
    per_atom=len(mol.ao_labels())//(2*n)
    imp=[i*per_atom+spec.get('imp_ao',0) for i in range(n)]
    d=aodmet.AODMET(mf,imp_idx=imp,threshold=spec.get("threshold",1e-10),verbose=0)
    d.build(save_chk=False)
    ldm=d.lowdin_orth()[0];env=[i for i in range(mol.nao) if i not in d.imp_idx]
    occ=np.linalg.eigvalsh(ldm[np.ix_(env,env)])
    bath_mask=((occ>=d.threshold)&(occ<=2-d.threshold2))|(occ>=2+d.threshold2)
    core_mask=(occ>2-d.threshold2)&(occ<2+d.threshold2)
    bath=occ[bath_mask]
    ext=occ[(occ<d.threshold)|core_mask]
    groups=outer.group_indexes_by_diff(bath,threshold=1e-8)[1]
    gap=float(np.min(np.abs(bath[:,None]-ext[None,:]))) if len(ext) else None
    neo=int(d.es_mf.mo_coeff.shape[1]); nelec=int(d.es_mf.mol.nelectron)
    if neo>10: raise RuntimeError(f'Unexpectedly large active space: {neo} orbitals; bath={bath.tolist()}, full={occ.tolist()}')
    metadata=dict(bath_occupations=bath.tolist(),bath_count=len(bath),
        bath_spread=float(np.ptp(bath)) if len(bath) else None,
        external_gap=gap,ncore=int(sum(core_mask)), ncas=neo,nelec=nelec,
        degenerate_count=int(sum(len(g) for g in groups)),
        degenerate_groups=[np.asarray(g).tolist() for g in groups])
    if energy_only:
        mc=tight_casscf(d.es_mf,neo,nelec)
        mc.conv_tol=1e-13;mc.conv_tol_grad=1e-8;mc.fcisolver.conv_tol=1e-14
        mc.kernel()
        if not mc.converged: raise RuntimeError('FCI/CASSCF not converged')
        return dict(energy=float(mc.e_tot+d.fo_ene), **metadata)
    grad=d.Gradients();grad.embedding_response=mode
    grad.zvector_solver='gmres';grad.zvector_warm_start=False;grad.zvector_recycle=False
    grad.zvector_cache_static=True;grad.zvector_reuse_residual=False
    grad._krylov_conv_tol=1e-10;grad.zvector_max_cycle=400
    if legacy_damping is not None: grad._perturbation=legacy_damping
    grad.solver_option=dict(solver='CASSCF',ncas=neo,nelec=nelec,fix_spin=False,
                            conv_tol=1e-14,conv_tol_grad=1e-10,max_cycle=100)
    t=time.perf_counter()
    with patch.object(solvers.mcscf,"CASSCF",tight_casscf):
        g,e=grad.kernel()
    elapsed=time.perf_counter()-t
    if not grad.es_solver.converged: raise RuntimeError('FCI/CASSCF not converged')
    if not np.all(np.isfinite(g)): raise RuntimeError('Nonfinite gradient')
    return dict(energy=float(e),gradient=g.tolist(),seconds=elapsed,
                zvector=grad.zvector_stats,**metadata)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=['h2','lif_core','lif_bond'],default='h2')
    parser.add_argument('--bond',type=float)
    parser.add_argument('--fragments',type=int,default=4)
    args=parser.parse_args()
    spec=({'molecule':'H2','basis':'6-31g','bond':1.4,'imp_ao':0} if args.case=='h2' else
          {'molecule':'LiF','basis':'sto-3g','bond':3.,'imp_ao':0 if args.case=='lif_core' else 4})
    if args.bond is not None:spec['bond']=args.bond
    spec['fragments']=args.fragments
    spec['threshold']=1e-10
    spec['scf_conv_tol']=1e-14
    spec['scf_conv_tol_grad']=1e-10
    spec['fci_conv_tol']=1e-14
    result=dict(spec=spec)
    output=Path(__file__).with_name(f'degenerate_correlated_{args.case}_{spec["bond"]}_{args.fragments}.json')
    try:
        for mode in ('legacy','subspace'):
            result[mode]=evaluate(spec,mode=mode)
            print(mode,result[mode],flush=True)
            output.write_text(json.dumps(result,indent=2))
        result['finite_differences']=[]
        for axis in (2,0):
            for h in (1e-3,5e-4):
                ep=evaluate(spec,h,energy_only=True,axis=axis)
                em=evaluate(spec,-h,energy_only=True,axis=axis)
                numerical=(ep['energy']-em['energy'])/(2*h)
                row=dict(axis=axis,step=h,numerical=numerical,
                    legacy_error=abs(numerical-result['legacy']['gradient'][0][axis]),
                    subspace_error=abs(numerical-result['subspace']['gradient'][0][axis]),
                    plus=ep,minus=em)
                result['finite_differences'].append(row)
                print('FD',row,flush=True)
                output.write_text(json.dumps(result,indent=2))
        result['gradient_max_difference']=float(np.max(np.abs(np.array(result['legacy']['gradient'])-result['subspace']['gradient'])))
        assert result['subspace']['degenerate_count']>=2
        assert abs(result['legacy']['energy']-result['subspace']['energy'])<1e-10
        result['completed']=True
    except Exception as exc:
        result['failure']=repr(exc)
        raise
    finally:
        output.write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
