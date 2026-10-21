import sys, os
from pyscf import scf, gto, lib, tools
sys.path.insert(0, '/lustre/home/2501110357/04grad/dmet_grad_pt2')
from embed_sim import rdiis, aodmet, ssdmet, myavas, sacasscf_mixer
from embed_sim.grad import aodmet as aodmet_grad
from embed_sim.grad import ssdmet as ssdmet_grad
from embed_sim.grad.guess import CASGuessCache
def get_mol(xyz_path, basis, charge, spin, max_memory=180000, verbose=4):
    mol = gto.Mole()
    mol.atom = xyz_path
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.verbose = verbose
    mol.max_memory = max_memory
    mol.build()
    return mol


### total input lines ### 
# here we should specify the input files # 
## mol setting ## 
## input files all in one 
def which_A(name, inp_spin):
    if name == 'a1':
        if inp_spin == 'ls':
            xyzpath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/sse17_struc/a1_2.xyz'
            spin = 1
            chkfilepath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/05a1/clustera1_rohf_r_M_cas_full_rohf.chk'
            
        elif inp_spin == 'hs':
            xyzpath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/sse17_struc/a1_6.xyz'
            spin = 5
            chkfilepath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/05a1/clustera1_rohf_r_M_cas_full_hs_rohf.chk'
        charge = 1
        basis = {'default': 'ccpvdz-dk', 'Fe': 'ccpwCVTZ-dk', 'O': 'ccpvtz-dk', 'N': 'ccpvtz-dk' }
        core1 = ['Fe 3d', 'Fe 4d']
        
    elif name == 'a2':
        if inp_spin == 'ls':
            xyzpath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/sse17_struc/a2_1.xyz'
            spin = 0
            chkfilepath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/05a2/clustera2_rohf_r_M_12o_ae_rohf.chk'
            
        elif inp_spin == 'hs':
            xyzpath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/sse17_struc/a2_5.xyz'
            spin = 4
            chkfilepath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/05a2/clustera2_rohf_r_M_12o_ae_hs_rohf.chk'
        charge = 0
        basis = {'default': 'ccpvdz-dk', 'Fe': 'ccpwCVTZ-dk', 'N1': 'ccpvtz-dk' }
        core1 = ['Fe 3d', 'Fe 4d']
    elif name == 'a3':
        if inp_spin == 'ls':
            xyzpath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/sse17_struc/a3_1.xyz'
            spin = 0
            chkfilepath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/05a3/clustera3_rohf_r_ae_rohf.chk'
            
        elif inp_spin == 'hs':
            xyzpath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/sse17_struc/a3_5.xyz'
            spin = 4
            chkfilepath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/05a3/clustera3_rohf_r_ae_hs_rohf.chk'
        charge = 0
        basis ={'default': 'ccpvdz-dk', 'Fe': 'ccpwCVTZ-dk', 'N1': 'ccpvtz-dk' }
        core1 = ['Fe 3d', 'Fe 4d']
    elif name == 'a4':
        if inp_spin == 'ls':
            xyzpath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/sse17_struc/a4_1.xyz'
            spin = 0
            chkfilepath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/05a4/a4_rohf_r_M_4_4L_ae3_ls_rohf.chk'
            
        elif inp_spin == 'hs':
            xyzpath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/sse17_struc/a4_5.xyz'
            spin = 4
            chkfilepath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/05a4/a4_rohf_r_M_4_4L_ae3_hs_rohf.chk'
        charge = 2
        basis = {'default': 'ccpvdz-dk', 'Fe': 'ccpwCVTZ-dk', 'N': 'ccpvtz-dk' }

        core1 = ['Fe 3d', 'Fe 4d']
    elif name == 'a5':
        if inp_spin == 'ls':
            xyzpath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/sse17_struc/a5_1.xyz'
            spin = 0
            chkfilepath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/05a5/clustera5_rohf_r_M_ae_rohf.chk'
            
        elif inp_spin == 'hs':
            xyzpath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/sse17_struc/a5_5.xyz'
            spin = 4
            chkfilepath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/05a5/clustera5_rohf_r_M_ae_hs_rohf.chk'
        charge = 2
        basis = {'default': 'ccpvdz-dk', 'Fe': 'ccpwCVTZ-dk', 'N1': 'ccpvtz-dk' }

        core1 = ['Fe 3d', 'Fe 4d']
    elif name == 'a6':
        if inp_spin == 'ls':
            xyzpath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/sse17_struc/a6_1.xyz'
            spin = 0
            chkfilepath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/05a6/a6_rohf_r_M_cons_12o_1L_full11_ls_rohf.chk'
            
        elif inp_spin == 'hs':
            xyzpath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/sse17_struc/a6_5.xyz'
            spin = 4
            chkfilepath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/05a6/a6_rohf_r_M_cons_12o_1L_full11_hs_rohf.chk'
        charge = -1
        basis = {'default': 'ccpvdz-dk', 'Fe': 'ccpwCVTZ-dk', 'N1': 'ccpvtz-dk' , 'C1': 'ccpvtz-dk'}

        core1 = ['Fe 3d', 'Fe 4d']
    elif name == 'a7':
        if inp_spin == 'ls':
            xyzpath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/sse17_struc/a7_2.xyz'
            spin = 1
            chkfilepath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/05a7/clustera7_rohf_r_M_ae_rohf.chk'
            
        elif inp_spin == 'hs':
            xyzpath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/sse17_struc/a7_4.xyz'
            spin = 3
            chkfilepath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/05a7/clustera7_rohf_r_M_ae_hs_rohf.chk'
        charge = 2
        basis ={'default': 'ccpvdz-dk', 'Co': 'ccpwCVTZ-dk', 'N': 'ccpvtz-dk' }

        core1 = ['Co 3d', 'Co 4d']
    elif name == 'a8':
        if inp_spin == 'ls':
            xyzpath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/sse17_struc/a8_1.xyz'
            spin = 0
            chkfilepath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/05a8/clustera8_rohf_r_M_12o_rohf.chk'
            
        elif inp_spin == 'hs':
            xyzpath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/sse17_struc/a8_3.xyz'
            spin = 2
            chkfilepath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/05a8/clustera8_rohf_r_M_12o_hs_rohf.chk'
        charge = 0
        basis ={'default': 'ccpvdz-dk', 'Ni': 'ccpwCVTZ-dk', 'O': 'ccpvtz-dk' , 'C1': 'ccpvtz-dk'}
        core1 = ['Ni 3d', 'Ni 4d']
    elif name == 'a9':
        if inp_spin == 'ls':
            xyzpath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/sse17_struc/a9_2.xyz'
            spin = 1
            chkfilepath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/05a9/a9_rohf_r_ML_10o_ae5_ls_rohf.chk'
            
        elif inp_spin == 'hs':
            xyzpath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/sse17_struc/a9_6.xyz'
            spin = 5
            chkfilepath = '/lustre/home/2501110357/01sco/04SSE17/07CAHF/05a9/a9_rohf_r_ML_10o_ae5_hs_rohf.chk'
        charge = 0
        basis = {'default': 'ccpvdz-dk', 'Mn': 'ccpwCVTZ-dk', 'C': 'ccpvtz-dk' }

        core1 = ['Mn 3d', 'Mn 4d']


    else:
        raise ValueError(f"Unknown name {name} or spin {spin}")
    return xyzpath, basis, charge, spin, core1, chkfilepath
xyzpath, basis, charge, spin, core1, chkfilepath = which_A('a1', 'ls')
xyzpath = xyzpath
basis = basis
charge = charge
spin = spin
max_memory = 180000

mol = get_mol(xyzpath, basis, charge, spin, max_memory=max_memory, verbose=4)
chkfilepath = chkfilepath

## casscf setting 

core1 = core1 ### avas setting 

### dmet input setting 
aodmet_yes = False
### specify the input files ends ### 
### need to change ### 
NCAS = 10
NELEC = 9
minao = mol._basis['Fe'] ### avas setting
statelis = [0, 3, 0, 0, 0, 0] ### number of singlets, doublets, triplets, quartets, quintets, sextets
imp = mol.search_ao_label(['Fe.*','O.*p', 'N.*p'])


current_file_name = os.path.basename(sys.argv[0])

title = f"{current_file_name.split('.')[0]}"


mf = scf.rohf.ROHF(mol).x2c()
mf.init_guess = 'atom'
mf.level_shift = 2.0
mf.conv_tol = 1e-8
mf.max_memory =getattr(mol, 'max_memory', 200000)
#mf.max_memory = 200000
mf.max_cycle = 250
mf.chkfile = '{0}_rohf.chk'.format(title)
t0 = lib.logger.perf_counter()
scfdat = scf.chkfile.load(chkfilepath, 'scf')

mf.e_tot = scfdat['e_tot']
mf.mo_coeff = scfdat['mo_coeff']
mf.mo_occ = scfdat['mo_occ']
mf.mo_energy = scfdat['mo_energy']
dm = mf.make_rdm1()

mf.kernel(dm0=dm)


mf_scanner = mf.as_scanner()
t3 = lib.logger.perf_counter()
evaluation = 0
cas_guess_cache = CASGuessCache()  # Persist across cas_grad calls.

def cas_grad(mol):
    mf_scanner(mol)
    global evaluation
    if aodmet_yes:
        mydmet = aodmet.AODMET(mf_scanner, title = title, imp_idx=imp, threshold= 1e-12, verbose=4)
    else:
        mydmet = ssdmet.SSDMET(mf_scanner,title = title,  imp_idx=imp, threshold= 1e-12, verbose=4)
    mydmet.build(save_chk=False)

    if not cas_guess_cache.has_guess:
        ncas, nelec, mo = mydmet.avas_export(
            core1, minao=minao, threshold=0.5, openshell_option=2)
        print(f"Initial AVAS: CAS({nelec},{ncas})")
        if (ncas, nelec) != (NCAS, NELEC):
            raise RuntimeError(
                f'AVAS selected CAS({nelec},{ncas}); expected CAS({NELEC},{NCAS}). '
                'Inspect the active orbitals before changing the selection.'
            )
        
        print(f"read from converged")
        
    else:
        ncas, nelec, mo = NCAS, NELEC, None
        print("Reusing previous converged SA-CASSCF MO/CI as NEVPT2 reference guess.",
              flush=True)
    grad = mydmet.Gradients()
    grad.solver_option = {
        "solver": "NEVPT2",
        "ncas": ncas,
        "nelec": nelec,
        "statelis": statelis,
        "init_guess": mo,
        "guess_cache": cas_guess_cache,
        "state": 0
    }
    de, e = grad.kernel()
    evaluation += 1
    with open(f'{title}_traj.xyz', 'a') as handle:
        handle.write(mol.tostring(format='xyz') + '\n')
    with open(f'{title}_energies.dat', 'a') as handle:
        handle.write(f'{evaluation:5d} {e:.14f}\n')
    print(f'NEVPT2 evaluation {evaluation}: E = {e:.14f}', flush=True)

    
    return e, de
from pyscf.geomopt import berny_solver, geometric_solver, as_pyscf_method
method = as_pyscf_method(mol, cas_grad)
new_mol = geometric_solver.optimize(method)
print("old geom")
print(mol.atom_coords())

print("new geom")
print(new_mol.atom_coords())
mol.tofile(f"{title}_init.xyz", format = 'xyz')
new_mol.tofile(f"{title}_opt.xyz", format = 'xyz')

mf = scf.rohf.ROHF(new_mol).x2c().density_fit()
mf.init_guess = 'atom'
mf.level_shift = 2.0
mf.conv_tol = 1e-8
mf.max_memory =getattr(mol, 'max_memory', 200000)
#mf.max_memory = 200000
mf.max_cycle = 250
mf.chkfile = '{0}_opt_rohf.chk'.format(title)
mf.kernel()
dm = mf.make_rdm1()

mf.level_shift = 1.0
mf.kernel(dm)
dm = mf.make_rdm1()
mf.level_shift = 0.0

mf.kernel(dm)


ncas, nelec, mo = myavas.avas(mf, core1, minao = mol._basis['Fe'], threshold=0.5, openshell_option=2)
#if ncas != 10:
#    sys.exit()
if ncas != 12:
    sys.exit('ncas is not 12, please check the impurity orbitals!')
tools.molden.from_mo(mol, f"{title}_avas.molden", mo)
tools.molden.from_scf(mf, f"{title}_scf.molden")

mycas = sacasscf_mixer.sacasscf_mixer(mf, ncas, nelec, statelis = statelis)
mycas.chkfile = f"{title}_cas.chk"
mycas.ah_level_shift = 1e-4
mycas.kernel(mo)
ecorr = sacasscf_mixer.sacasscf_nevpt2(mycas, method='SC')
print(f"cas energy",mycas.fcisolver.e_states )
print(f"NEVPT2 energy",ecorr)
mycas.fcisolver.e_states = mycas.fcisolver.e_states + ecorr
print(f"SACASSCF+NEVPT2 energies: {mycas.fcisolver.e_states}")
mycas.analyze()
tools.molden.from_mcscf(mycas, f"{title}.molden", cas_natorb = True)


