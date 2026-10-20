"""Dense, fully analytic single-state IC-CASPT2 nuclear gradients.

Reference implementation for small real, all-electron CASSCF systems. Nuclear
integral derivatives are supplied by libcint. CASSCF response is obtained from
the analytic orbital/CI Hessian; the existing CASPT2 tensor equations are
evaluated using first-order dual arithmetic. No finite differences are used.

Limitations: no frozen orbitals, state averaging, density fitting, ECP, IPEA
shift, or out-of-core storage. The dense Hessian and amplitude matrix make this
an intentionally expensive correctness reference rather than a production
implementation. A fixed overlap-truncation rank is required.
"""
import copy
import types
import numpy as np
from pyscf import ao2mo, fci, lib
from pyscf.grad import rhf
from pyscf.lib import logger
from pyscf.mcscf import newton_casscf
from . import caspt2
from ._dual import Dual, Numpy, dual, einsum, eigh


def _tensor_functions():
    """Private function globals: do not monkey-patch the energy implementation."""
    namespace = dict(vars(caspt2), np=Numpy(), _einsum=einsum, disjoint_eigh=eigh)
    names = ('compute_lhs_amplitudes', 'compute_rhs_amplitudes',
             'basis_orthogonalize', 'amplitudes_to_vector', 'vector_to_amplitudes')
    for name in names:
        fn = getattr(caspt2, name)
        namespace[name] = types.FunctionType(fn.__code__, namespace, name, fn.__defaults__)
    return types.SimpleNamespace(**{name: namespace[name] for name in names})


def _rdms(ci, ncas, nelec, dci=None):
    def transition(bra, ket):
        if sum(nelec) > 1:
            ds = fci.rdm.make_dm1234('FCI4pdm_kern_sf', bra, ket, ncas, nelec)
            ds = fci.rdm.reorder_dm1234(*ds)
        else:
            ds = fci.rdm.make_dm123('FCI3pdm_kern_sf', bra, ket, ncas, nelec)
            ds = (*fci.rdm.reorder_dm123(*ds), np.zeros((ncas,) * 8))
        return [d.transpose(*range(0, d.ndim, 2), *range(1, d.ndim, 2))
                for d in ds]
    base = transition(ci, ci)
    if dci is None:
        return base
    left, right = transition(dci, ci), transition(ci, dci)
    return [Dual(v, l + r) for v, l, r in zip(base, left, right)]


def _transform(g, u):
    # Sequential contractions avoid an eight-index temporary.
    for _ in range(4):
        g = Numpy().tensordot(u.T, g, axes=(1, 0)).transpose(1, 2, 3, 0)
    return g


def _effective(h, g, dm1, nc, na):
    a = slice(nc, nc + na)
    h_eff = h.copy()
    for i in range(nc):
        h_eff = h_eff + 2 * g[:, :, i, i] - g[:, i, i, :]
    fock = h_eff + einsum(None, 'pqrs,rs->pq', g[:, :, a, a], dm1)
    fock = fock - .5 * einsum(None, 'prsq,rs->pq', g[:, a, a, :], dm1)
    return h_eff, fock


def _correlation(h, g, rdms, nc, na, threshold, functions):
    """Differentiate a dense linear solve through its stationary functional."""
    n = h.shape[0]
    nv = n - nc - na
    h_eff, fock = _effective(h, g, rdms[0], nc, na)
    # Canonicalize core and external blocks, including their analytic response.
    u = Dual(np.eye(n))
    for block in (slice(0, nc), slice(nc + na, n)):
        if fock[block, block].size:
            w, v = eigh(fock[block, block])
            u[block, block] = v
    h_eff = u.T @ h_eff @ u
    fock = u.T @ fock @ u
    g = _transform(g, u)
    a, c, v = slice(nc, nc + na), slice(0, nc), slice(nc + na, n)
    eris = (g[:, :, a, a].transpose(0, 2, 1, 3),
            g[:, a, :, a].transpose(0, 2, 1, 3),
            g[:, a, c, v].transpose(0, 2, 1, 3),
            g[c, v, c, v].transpose(0, 2, 1, 3))
    mats, us, shapes = functions.basis_orthogonalize(nc, na, nv, rdms, threshold)
    rhs = functions.compute_rhs_amplitudes(nc, na, nv, rdms, h_eff, eris, 'numpy')
    b = functions.amplitudes_to_vector(nc, na, nv, rhs, us)
    if not b.size:
        return Dual(0.)
    if b.size > 1000:
        raise NotImplementedError('Dense reference limited to 1000 contracted amplitudes')
    # Fix the coordinate vector when differentiating the stationary functional;
    # U and the physical amplitudes still carry their metric derivatives.
    matrix, dmatrix = [], []
    for col in np.eye(b.size):
        ts = functions.vector_to_amplitudes(nc, na, nv, Dual(col), us, shapes)
        ats = functions.compute_lhs_amplitudes(nc, na, nv, ts, rdms, fock,
                                               'numpy', 4000)
        av = functions.amplitudes_to_vector(nc, na, nv, ats, us)
        matrix.append(av.v)
        dmatrix.append(av.d)
    matrix, dmatrix = np.array(matrix).T, np.array(dmatrix).T
    if np.max(np.abs(matrix - matrix.T)) > 1e-8:
        raise RuntimeError('CASPT2 amplitude matrix is not symmetric')
    x = np.linalg.solve(matrix, b.v)
    return Dual(x @ b.v, 2 * x @ b.d - x @ dmatrix @ x)


def _eris(g, nc, na):
    a = slice(nc, nc + na)
    n = g.shape[0]
    vc = np.zeros((n, n))
    for i in range(nc):
        vc += 2 * g[:, :, i, i] - g[:, i, i, :]
    return types.SimpleNamespace(
        ppaa=g[:, :, a, a], papa=g[:, a, :, a], vhf_c=vc,
        j_pc=np.einsum('ppii->pi', g[:, :, :nc, :nc]),
        k_pc=np.einsum('piip->pi', g[:, :nc, :nc, :]))


def _response_setup(mc, h, g, ci):
    # gen_g_hop uses MO integrals; identity MOs allow exact integral perturbations
    # to enter the same analytic CASSCF gradient without AO back-transforms.
    work = copy.copy(mc)
    n = h.shape[0]
    work.get_hcore = lambda *args: h
    # The orbital Hessian also invokes update_jk_in_ah. Its density matrices
    # are in this synthetic MO basis, so its Coulomb/exchange operator must use
    # MO integrals rather than the original AO get_jk implementation.
    def get_jk(mol, dm):
        return (np.einsum('pqrs,...rs->...pq', g, dm),
                np.einsum('prsq,...rs->...pq', g, dm))
    work.get_jk = get_jk
    gg, _, hop, _ = newton_casscf.gen_g_hop(work, np.eye(n), ci, _eris(g, mc.ncore, mc.ncas))
    no = mc.pack_uniq_var(np.zeros((n, n))).size
    hessian = np.column_stack([hop(e) for e in np.eye(gg.size)])
    asym = np.max(np.abs(hessian - hessian.T))
    if asym > 1e-7:
        raise RuntimeError('CASSCF Hessian is not symmetric: %g' % asym)
    w, u = np.linalg.eigh((hessian + hessian.T) / 2)
    keep = np.abs(w) > 1e-8
    pinv = (u[:, keep] / w[keep]) @ u[:, keep].T
    if np.linalg.norm(gg) > 2e-6:
        raise RuntimeError('CASSCF reference is not sufficiently stationary: %g' % np.linalg.norm(gg))

    def response(dh, dg):
        work.get_hcore = lambda *args: dh
        b = newton_casscf.gen_g_hop(work, np.eye(n), ci, _eris(dg, mc.ncore, mc.ncas))[0]
        z = -pinv @ b
        residual = np.linalg.norm(hessian @ z + b)
        if residual > 2e-7:
            raise RuntimeError('CASSCF response residual %g' % residual)
        return mc.unpack_uniq_var(z[:no]), z[no:].reshape(ci.shape), residual
    return response


def _rotate_derivative(g, k):
    return (np.einsum('ip,iqrs->pqrs', k, g) + np.einsum('jq,pjrs->pqrs', k, g)
            + np.einsum('kr,pqks->pqrs', k, g) + np.einsum('ls,pqrl->pqrs', k, g))


class Gradients(rhf.GradientsBase):
    """Small-system analytic reference; see module limitations."""
    def __init__(self, method):
        self.base = method
        self.mol = method._mc.mol
        self.verbose = method.verbose
        self.stdout = self.mol.stdout
        self.max_memory = method._mc.max_memory
        self.atmlst = None
        self.de = None
        self.response_residuals = []
        self.reference_energy_error = None

    def kernel(self, atmlst=None):
        mr, mc = self.base, self.base._mc
        if not hasattr(mc, 'uniq_var_indices') or mc.ncas == 0:
            raise NotImplementedError('Requires CASSCF with a nonempty active space')
        if (mr.frozen or mr.ipea_shift or mr.root or mc.fcisolver.nroots != 1
                or getattr(mc, 'weights', None) is not None
                or getattr(mc, 'with_df', None) or mc.frozen
                or self.mol.has_ecp() or self.mol.symmetry):
            raise NotImplementedError('Requires single-state, real, all-electron, unfrozen '
                                      'CASSCF without DF, symmetry, or IPEA shift')
        if not getattr(mc, 'converged', False):
            raise RuntimeError('CASSCF is not converged')
        if np.iscomplexobj(mc.mo_coeff) or np.iscomplexobj(mc.ci):
            raise NotImplementedError('Complex reference')
        # This implementation intentionally bounds allocations for a reference solver.
        if mc.mo_coeff.shape[1] > 16 or np.asarray(mc.ci).size > 100:
            raise NotImplementedError('Dense reference limited to 16 MOs and 100 CI coefficients')
        mol, mo, ci = self.mol, mc.mo_coeff, np.asarray(mc.ci)
        n = mo.shape[1]
        h = mo.T @ mc.get_hcore() @ mo
        g = ao2mo.kernel(mol, mo, compact=False).reshape((n,) * 4)
        response = _response_setup(mc, h, g, ci)
        functions = _tensor_functions()
        base_rdms = _rdms(ci, mc.ncas, mc.nelecas)
        reference = _correlation(Dual(h), Dual(g), [Dual(d) for d in base_rdms],
                                 mc.ncore, mc.ncas, mr.ortho_thrds, functions)
        if not getattr(mr, 'converged', False):
            mr.kernel()
        if not mr.converged:
            raise RuntimeError('CASPT2 amplitudes are not converged')
        self.reference_energy_error = float(reference.v - mr.e_corr)
        if abs(self.reference_energy_error) > 1e-8:
            raise RuntimeError('Dense CASPT2 does not reproduce energy: %g' % self.reference_energy_error)
        mfgrad = mc._scf.nuc_grad_method()
        hgen = mfgrad.hcore_generator(mol)
        ip_s = mol.intor('int1e_ipovlp', comp=3)
        ip_g = mol.intor('int2e_ip1', comp=3, aosym='s1')
        aoslices = mol.aoslice_by_atom()
        atoms = list(range(mol.natm)) if atmlst is None else list(atmlst)
        self.atmlst = atoms
        # Reference CASSCF gradient is already fully relaxed.
        result = mc.nuc_grad_method().kernel(atmlst=atoms).copy()
        self.response_residuals = []
        for row, atom in enumerate(atoms):
            p0, p1 = aoslices[atom, 2:]
            for xyz in range(3):
                ds = np.zeros((mol.nao, mol.nao))
                ds[p0:p1] -= ip_s[xyz, p0:p1]
                ds[:, p0:p1] -= ip_s[xyz, p0:p1].T
                conn = -.5 * (mo.T @ ds @ mo)
                dh = mo.T @ hgen(atom)[xyz] @ mo + conn.T @ h + h @ conn
                dg_ao = np.zeros((mol.nao,) * 4)
                first = np.zeros_like(dg_ao)
                first[p0:p1] = -ip_g[xyz, p0:p1]
                dg_ao = first + first.transpose(1, 0, 2, 3)
                dg_ao += first.transpose(2, 3, 0, 1) + first.transpose(2, 3, 1, 0)
                dg = ao2mo.incore.full(dg_ao, mo, compact=False).reshape((n,) * 4)
                dg += _rotate_derivative(g, conn)
                k, dci, residual = response(dh, dg)
                dh += k.T @ h + h @ k
                dg += _rotate_derivative(g, k)
                rdms = _rdms(ci, mc.ncas, mc.nelecas, dci)
                ec = _correlation(Dual(h, dh), Dual(g, dg), rdms, mc.ncore,
                                  mc.ncas, mr.ortho_thrds, functions)
                result[row, xyz] += float(ec.d)
                self.response_residuals.append(residual)
                logger.info(self, 'CASPT2 gradient atom %d axis %d: %.12g', atom, xyz, result[row, xyz])
        self.de = result
        return result

    grad = kernel

    def as_scanner(self):
        if isinstance(self, lib.GradScanner):
            return self
        return _GradScanner(self)


class _GradScanner(lib.GradScanner, Gradients):
    """Energy/gradient scanner accepted by PySCF geometry optimizers."""
    def __init__(self, gradient):
        self.__dict__.update(gradient.__dict__)
        self.base = copy.copy(gradient.base)
        self.base._mc = gradient.base._mc.as_scanner()

    def __call__(self, mol_or_geom, **kwargs):
        from pyscf import gto
        mol = (mol_or_geom if isinstance(mol_or_geom, gto.MoleBase)
               else self.mol.set_geom_(mol_or_geom, inplace=False))
        self.base._mc(mol)
        self.mol = mol
        self.base.kernel()
        gradient = self.kernel(**kwargs)
        return self.base.e_tot, gradient
