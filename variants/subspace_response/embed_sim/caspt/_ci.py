"""Contract normal-ordered spin-free RDM weights directly with a CI vector.

The annihilated-CI representation groups spin sectors and antisymmetric orbital
tuples. It never enumerates CI unit vectors or constructs transition 4-RDMs.
Only one compressed spin block is kept at a time.
"""
from itertools import combinations, permutations, product
from math import comb
import numpy as np
from scipy import sparse
from pyscf.fci import cistring


def _parity(perm):
    return (-1) ** sum(perm[i] > perm[j] for i in range(len(perm))
                       for j in range(i+1, len(perm)))


def _permutations(na, nb):
    for a in permutations(range(na)):
        for b in permutations(range(nb)):
            yield a + tuple(na+i for i in b), _parity(a)*_parity(b)


def _des_map(norb, nelec, orbitals):
    strings = cistring.make_strings(range(norb), nelec)
    addresses, targets, signs = [], [], []
    for addr, raw in enumerate(strings):
        string, sign = int(raw), 1
        for orbital in orbitals:
            if not string & (1 << orbital):
                sign = 0
                break
            sign *= cistring.des_sign(orbital, string)
            string ^= 1 << orbital
        if sign:
            addresses.append(addr)
            targets.append(string)
            signs.append(sign)
    targets = cistring.strs2addr(norb, nelec-len(orbitals), np.asarray(targets, dtype=np.int64))
    return np.asarray(addresses), targets, np.asarray(signs)


def _annihilator(norb, nelec, na, nb, tuples):
    nea, neb = nelec
    nsa, nsb = comb(norb, nea), comb(norb, neb)
    nta, ntb = comb(norb, nea-na), comb(norb, neb-nb)
    rows, cols, data = [], [], []
    amap, bmap = {}, {}
    for number, orbitals in enumerate(tuples):
        alpha, beta = tuple(orbitals[:na]), tuple(orbitals[na:])
        if alpha not in amap:
            amap[alpha] = _des_map(norb, nea, alpha)
        if beta not in bmap:
            bmap[beta] = _des_map(norb, neb, beta)
        ia, ja, sa = amap[alpha]
        ib, jb, sb = bmap[beta]
        rows.append((number*nta*ntb + ja[:, None]*ntb+jb[None, :]).ravel())
        cols.append((ia[:, None]*nsb+ib[None, :]).ravel())
        # The alpha/beta crossing parity is constant for this whole spin
        # block and cancels between the annihilator and its adjoint.
        data.append((sa[:, None]*sb[None, :]).ravel())
    return sparse.csr_matrix((np.concatenate(data),
                              (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(tuples)*nta*ntb, nsa*nsb)), nta*ntb


def _compressed_weight(weight, norb, rank, na, nb, tuples):
    result = np.zeros((len(tuples), len(tuples)))
    matrix = weight.reshape(norb**rank, norb**rank)
    orbital_perms = list(_permutations(na, nb))
    for spins in product((0, 1), repeat=rank):
        if spins.count(0) != na:
            continue
        group = [i for i, spin in enumerate(spins) if spin == 0]
        group += [i for i, spin in enumerate(spins) if spin == 1]
        inverse = np.argsort(group)
        indices = []
        for perm, sign in orbital_perms:
            ordered = tuples[:, perm][:, inverse]
            flat = np.ravel_multi_index(tuple(ordered.T), (norb,)*rank)
            indices.append((flat, sign))
        for left, sl in indices:
            for right, sr in indices:
                result += sl*sr*matrix[np.ix_(left, right)]
    return result


def contract_rdm_weights(weights, ci, norb, nelec, max_memory=2000):
    """Return d(sum_k W_k:Gamma_k)/dCI without transition RDMs.

    Gamma uses grouped creation/annihilation indices, as in CASPT2. The real
    1-RDM convention is transposed relative to higher ranks, which has no
    effect after the required Hermitian symmetrization.
    """
    ci = np.asarray(ci)
    out = np.zeros(ci.size)
    diagnostics = dict(spin_blocks=0, transition_rdm_calls=0, max_block_bytes=0)
    for rank, weight in enumerate(weights, 1):
        if rank > sum(nelec) or not np.any(weight):
            continue
        for na in range(rank+1):
            nb = rank-na
            if na > nelec[0] or nb > nelec[1]:
                continue
            tuples = np.asarray([a+b for a in combinations(range(norb), na)
                                 for b in combinations(range(norb), nb)], dtype=int)
            count = len(tuples)
            residual = comb(norb, nelec[0]-na)*comb(norb, nelec[1]-nb)
            # Bound dense compressed weights, annihilated CI and sparse links.
            estimated = 8*(2*count**2+3*count*residual) + 16*count*ci.size
            if estimated > max_memory*1e6:
                raise MemoryError('Contracted CI block exceeds memory budget')
            op, residual = _annihilator(norb, nelec, na, nb, tuples)
            ket = (op @ ci.ravel()).reshape(count, residual)
            compressed = _compressed_weight(weight, norb, rank, na, nb, tuples)
            acted = (compressed + compressed.T) @ ket
            out += op.T @ acted.ravel()
            diagnostics['spin_blocks'] += 1
            diagnostics['max_block_bytes'] = max(diagnostics['max_block_bytes'],
                ket.nbytes+compressed.nbytes+acted.nbytes+op.data.nbytes+op.indices.nbytes+op.indptr.nbytes)
    return out.reshape(ci.shape), diagnostics
