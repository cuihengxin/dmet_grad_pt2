"""Matrix-free analytic single-state IC-CASPT2 gradients.

One reverse perturbation Lagrangian produces effective integral densities and
RDM weights. Contracted annihilation operators form the CI source; one CASSCF
Z-vector supplies relaxation. No full amplitude matrix or Hessian is built.

Scope: real, single-state all-electron CASSCF; no frozen orbitals, DF, IPEA or
ECP. Full conventional integrals and active RDMs still require N^4/a^8 memory.
"""
import copy
import time
import types
import numpy as np
from scipy.sparse.linalg import LinearOperator, minres
from pyscf import ao2mo, fci, lib
from pyscf.lib import logger
from pyscf.mcscf import newton_casscf
from . import caspt2
from . import reference
from ._reverse import Tensor, Numpy, einsum, eigh, value
from ._ci import contract_rdm_weights

_rdms = reference._rdms


def _tensor_functions():
    namespace = dict(vars(caspt2), np=Numpy(), _einsum=einsum, disjoint_eigh=eigh)
    names = ('compute_lhs_amplitudes', 'compute_rhs_amplitudes',
             'basis_orthogonalize', 'amplitudes_to_vector', 'vector_to_amplitudes')
    for name in names:
        fn = getattr(caspt2, name)
        namespace[name] = types.FunctionType(fn.__code__, namespace, name, fn.__defaults__)
    return types.SimpleNamespace(**{name: namespace[name] for name in names})


def _transform(g, u):
    for _ in range(4):
        g = einsum(None, 'ip,ijkl->jklp', u, g)
    return g


def _effective(h, g, nc):
    out = h.copy()
    for i in range(nc):
        out = out + 2*g[:, :, i, i] - g[:, i, i, :]
    return out


def _prepare(h, g, rdms, nc, na, threshold, functions):
    n, a = h.shape[0], slice(nc, nc+na)
    nv = n-nc-na
    h_eff = _effective(h, g, nc)
    fock = h_eff + einsum(None, 'pqrs,rs->pq', g[:, :, a, a], rdms[0])
    fock = fock - .5*einsum(None, 'prsq,rs->pq', g[:, a, a, :], rdms[0])
    u = Tensor(np.eye(n))
    for block in (slice(0, nc), slice(nc+na, n)):
        if fock[block, block].size:
            _, vectors = eigh(fock[block, block])
            u[block, block] = vectors
    h_eff, fock = u.T @ h_eff @ u, u.T @ fock @ u
    g = _transform(g, u)
    c, v = slice(0, nc), slice(nc+na, n)
    eris = (g[:, :, a, a].transpose(0, 2, 1, 3),
            g[:, a, :, a].transpose(0, 2, 1, 3),
            g[:, a, c, v].transpose(0, 2, 1, 3),
            g[c, v, c, v].transpose(0, 2, 1, 3))
    mats, us, shapes = functions.basis_orthogonalize(nc, na, nv, rdms, threshold)
    rhs = functions.compute_rhs_amplitudes(nc, na, nv, rdms, h_eff, eris, 'numpy')
    return fock, mats, us, shapes, functions.amplitudes_to_vector(nc, na, nv, rhs, us)


def _solve_amplitudes(nc, na, nv, fock, rdms, mats, us, shapes, b,
                      max_memory, max_cycle, tolerance):
    """Reuse energy kernels/preconditioner, storing only O(P) solver vectors."""
    calls = [0]
    def hop(x):
        calls[0] += 1
        ts = caspt2.vector_to_amplitudes(nc, na, nv, x, us, shapes)
        ats = caspt2.compute_lhs_amplitudes(nc, na, nv, ts, rdms, fock, 'numpy', max_memory)
        return caspt2.amplitudes_to_vector(nc, na, nv, ats, us)
    if not b.size or np.linalg.norm(b) < tolerance:
        return np.zeros_like(b), dict(iterations=0, operator_calls=0, residual=0.)
    es, rs = caspt2.compute_lhs_preconditioner(nc, na, nv, us, rdms, mats, 0., fock, 'numpy')
    def precondition(r):
        return caspt2.apply_lhs_preconditioner(nc, r, shapes, es, rs)
    x, r = np.zeros_like(b), b.copy()
    p = precondition(r)
    old = r @ p
    for iteration in range(1, max_cycle+1):
        z = hop(p)
        denominator = p @ z
        if not np.isfinite(denominator) or abs(denominator) < 1e-30:
            break
        alpha = old/denominator
        x += alpha*p
        r -= alpha*z
        if np.linalg.norm(r) <= tolerance:
            break
        z = precondition(r)
        new = r @ z
        p = z+(new/old)*p
        old = new
    residual = np.linalg.norm(hop(x)-b)
    if residual > max(tolerance*10, 1e-9):
        raise RuntimeError('CASPT2 matrix-free amplitude residual %g' % residual)
    return x, dict(iterations=iteration, operator_calls=calls[0], residual=float(residual))


def effective_densities(h, g, rdms, nc, na, threshold=1e-10,
                        max_memory=2000, max_cycle=200, tolerance=1e-11):
    """One reverse pass through stationary 2 x.b - x.Ax, without unrolling CG."""
    leaves = [Tensor(x, requires_grad=True) for x in [h, g, *rdms]]
    th, tg, *trdms = leaves
    functions = _tensor_functions()
    fock, mats, us, shapes, b = _prepare(th, tg, trdms, nc, na, threshold, functions)
    x, stats = _solve_amplitudes(nc, na, h.shape[0]-nc-na, value(fock), rdms,
        [value(m) for m in mats], [value(u) for u in us], shapes, value(b),
        max_memory, max_cycle, tolerance)
    energy = 0.
    if b.size:
        ts = functions.vector_to_amplitudes(nc, na, h.shape[0]-nc-na, Tensor(x), us, shapes)
        ats = functions.compute_lhs_amplitudes(nc, na, h.shape[0]-nc-na, ts,
            trdms, fock, 'numpy', max_memory)
        ax = functions.amplitudes_to_vector(nc, na, h.shape[0]-nc-na, ats, us)
        lagrangian = 2*einsum(None, 'i,i->', x, b)-einsum(None, 'i,i->', x, ax)
        lagrangian.backward()
        energy = float(lagrangian.v)
    stats.update(n_amplitudes=b.size, reverse_lhs_calls=int(bool(b.size)), amplitude_matrix_elements=0)
    return energy, [leaf.grad for leaf in leaves], stats


def _sym_integrals(dh, dg):
    dh = (dh+dh.T)/2
    dg = (dg+dg.transpose(1, 0, 2, 3)+dg.transpose(0, 1, 3, 2)
          +dg.transpose(1, 0, 3, 2))/4
    return dh, (dg+dg.transpose(2, 3, 0, 1))/2


def _orbital_derivative(h, g, dh, dg):
    dh, dg = _sym_integrals(dh, dg)
    return 2*h @ dh+4*np.einsum('pqrs,iqrs->ip', dg, g, optimize=True)


def _zvector(mc, mo, ci, source, tolerance=1e-10):
    eris = mc.ao2mo(mo)
    gradient, _, raw_hop, diagonal = newton_casscf.gen_g_hop(mc, mo, ci, eris)
    if np.linalg.norm(gradient) > 2e-6:
        raise RuntimeError('CASSCF reference not stationary: %g' % np.linalg.norm(gradient))
    no = source.size-ci.size
    ref = ci.ravel()/np.linalg.norm(ci)
    def project(x):
        x = x.copy()
        x[no:] -= ref*(ref @ x[no:])
        return x
    source = project(source)
    products, iterations = [0], [0]
    def hop(x):
        products[0] += 1
        px = project(x)
        return project(raw_hop(px)) + x-px
    operator = LinearOperator((len(source),)*2, matvec=hop, dtype=float)
    diagonal = np.maximum(np.abs(diagonal), 1e-6)
    preconditioner = LinearOperator(operator.shape, matvec=lambda x: x/diagonal, dtype=float)
    def callback(x): iterations[0] += 1
    if np.linalg.norm(source) < tolerance:
        z, info = np.zeros_like(source), 0
    else:
        z, info = minres(operator, -source, M=preconditioner, rtol=1e-12,
                         maxiter=max(200, min(2000, 4*len(source))), callback=callback)
    residual = np.linalg.norm(hop(z)+source)
    if info or residual > max(tolerance, 1e-8*np.linalg.norm(source)):
        raise RuntimeError('CASSCF Z-vector not converged: info=%s residual=%g' % (info, residual))
    z = project(z)
    return mc.unpack_uniq_var(z[:no]), z[no:].reshape(ci.shape), dict(
        zvector_solves=1, z_iterations=iterations[0], hessian_products=products[0],
        response_residual=float(residual), hessian_matrix_elements=0)


def _reference_energy(h, g, rdms, nc, na, include_core=True):
    energy = Tensor(0.)
    if include_core:
        for i in range(nc):
            energy = energy+2*h[i, i]
            for j in range(nc):
                energy = energy+2*g[i, i, j, j]-g[i, j, j, i]
    a = slice(nc, nc+na)
    energy = energy+einsum(None, 'pq,pq->', rdms[0], _effective(h, g, nc)[a, a])
    return energy+.5*einsum(None, 'prqs,pqrs->', rdms[1], g[a, a, a, a])


def _relaxation_densities(h, g, rdms, ci, kappa, zci, nc, na, nelec):
    # Two rank-1/2 transition calls for the single Z-vector, never rank 3/4.
    d1l, d2l = fci.direct_spin1.trans_rdm12(zci, ci, na, nelec)
    d1r, d2r = fci.direct_spin1.trans_rdm12(ci, zci, na, nelec)
    d_rdms = [d1l+d1r, (d2l+d2r).transpose(0, 2, 1, 3)]
    th, tg = Tensor(h, requires_grad=True), Tensor(g, requires_grad=True)
    ht = kappa.T @ th+th @ kappa
    gt = (einsum(None, 'ip,iqrs->pqrs', kappa, tg)
          +einsum(None, 'jq,pjrs->pqrs', kappa, tg)
          +einsum(None, 'kr,pqks->pqrs', kappa, tg)
          +einsum(None, 'ls,pqrl->pqrs', kappa, tg))
    mixed = (_reference_energy(ht, gt, rdms, nc, na)
             +_reference_energy(th, tg, d_rdms, nc, na, include_core=False))
    mixed.backward()
    return th.grad, tg.grad


def _density_to_ao(dg, mo):
    for _ in range(4):
        dg = np.tensordot(mo, dg, axes=(1, 0)).transpose(1, 2, 3, 0)
    return dg


class Gradients(reference.Gradients):
    """Reverse effective densities plus one matrix-free Z-vector."""
    def kernel(self, atmlst=None):
        t0 = time.perf_counter()
        mr, mc = self.base, self.base._mc
        if not hasattr(mc, 'uniq_var_indices') or mc.ncas == 0:
            raise NotImplementedError('Requires CASSCF with a nonempty active space')
        if (mr.frozen or mr.ipea_shift or mr.root or mc.fcisolver.nroots != 1
                or getattr(mc, 'weights', None) is not None
                or getattr(mc, 'with_df', None) or mc.frozen
                or self.mol.has_ecp() or self.mol.symmetry):
            raise NotImplementedError('Requires single-state unfrozen all-electron CASSCF '
                                      'without DF, symmetry or IPEA shift')
        if not mc.converged: raise RuntimeError('CASSCF is not converged')
        mol, mo, ci = self.mol, mc.mo_coeff, np.asarray(mc.ci)
        if np.iscomplexobj(mo) or np.iscomplexobj(ci):
            raise NotImplementedError('Complex reference')
        n, na, nc = mo.shape[1], mc.ncas, mc.ncore
        # Resource budgeting replaces arbitrary MO/CI/amplitude caps. Full
        # conventional integral and RDM storage is still explicitly bounded.
        available = max(0., self.max_memory-lib.current_memory()[0])
        estimated = 8*(48*max(n, mol.nao)**4+12*na**8)/1e6
        if estimated > available:
            raise MemoryError('Conventional gradient estimate %.0f MB exceeds available %.0f MB; '
                              'DF/blocking not yet implemented' % (estimated, available))
        if not getattr(mr, 'converged', False): mr.kernel()
        if not mr.converged: raise RuntimeError('CASPT2 amplitudes not converged')
        h = mo.T @ mc.get_hcore() @ mo
        g = ao2mo.kernel(mol, mo, compact=False).reshape((n,)*4)
        rdms = _rdms(ci, na, mc.nelecas)
        ec, densities, stats = effective_densities(h, g, rdms, nc, na,
            mr.ortho_thrds, available, max(200, mr.max_cycle), 1e-11)
        self.reference_energy_error = ec-mr.e_corr
        if abs(self.reference_energy_error) > 1e-8:
            raise RuntimeError('Matrix-free CASPT2 energy mismatch %g' % self.reference_energy_error)
        dh, dg, *weights = densities
        ci_source, ci_stats = contract_rdm_weights(weights, ci, na, mc.nelecas, max_memory=available)
        orbital = _orbital_derivative(h, g, dh, dg)
        source = np.concatenate([mc.pack_uniq_var(orbital-orbital.T), ci_source.ravel()])
        kappa, zci, z_stats = _zvector(mc, mo, ci, source)
        zh, zg = _relaxation_densities(h, g, rdms, ci, kappa, zci, nc, na, mc.nelecas)
        dh, dg = _sym_integrals(dh+zh, dg+zg)
        orbital = _orbital_derivative(h, g, dh, dg)
        d1ao = mo @ dh @ mo.T
        xao = mo @ ((orbital+orbital.T)/4) @ mo.T
        d2ao = _density_to_ao(dg, mo)
        atoms = list(range(mol.natm)) if atmlst is None else list(atmlst)
        self.atmlst = atoms
        result = mc.nuc_grad_method().kernel(atmlst=atoms).copy()
        mfgrad = mc._scf.nuc_grad_method()
        hgen = mfgrad.hcore_generator(mol)
        ip_s = mol.intor('int1e_ipovlp', comp=3)
        for row, atom in enumerate(atoms):
            sh0, sh1, p0, p1 = mol.aoslice_by_atom()[atom]
            derivative = mol.intor('int2e_ip1', comp=3, aosym='s1',
                shls_slice=(sh0, sh1, 0, mol.nbas, 0, mol.nbas, 0, mol.nbas))
            result[row] += np.einsum('xij,ij->x', hgen(atom), d1ao)
            result[row] -= 4*np.einsum('xijkl,ijkl->x', derivative, d2ao[p0:p1])
            result[row] += 2*np.einsum('xij,ij->x', ip_s[:, p0:p1], xao[p0:p1])
        self.de = result
        self.response_residuals = [z_stats['response_residual']]
        self.stats = dict(stats, **z_stats, ci=ci_stats, z_transition_rdm12_calls=2,
                          seconds=time.perf_counter()-t0)
        logger.info(self, 'CASPT2 reverse gradient: %s', self.stats)
        return result

    grad = kernel

    def as_scanner(self):
        if isinstance(self, lib.GradScanner): return self
        return _GradScanner(self)


class _GradScanner(lib.GradScanner, Gradients):
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
        return self.base.e_tot, self.kernel(**kwargs)
