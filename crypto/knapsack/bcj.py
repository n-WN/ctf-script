#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import ast
import math
import random
import time
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from itertools import combinations


@dataclass(frozen=True)
class Entry:
    sum_int: int
    sum_mod: int
    mask: int


@dataclass(frozen=True)
class EntryPN:
    sum_int: int
    sum_mod: int
    pos_mask: int
    neg_mask: int
    pos_count: int
    neg_count: int


def parse_set_arg(arg: str) -> List[int]:
    # If it's a file path, read ints from it
    if os.path.exists(arg):
        with open(arg, 'r', encoding='utf-8') as f:
            data = f.read()
        nums: List[int] = []
        token = ''
        for ch in data:
            if ch in '0123456789-':
                token += ch
            else:
                if token:
                    nums.append(int(token))
                    token = ''
        if token:
            nums.append(int(token))
        return nums
    # Try Python literal like "[1,2,3]"
    try:
        val = ast.literal_eval(arg)
        if isinstance(val, (list, tuple)):
            return [int(x) for x in val]
    except Exception:
        pass
    # Fallback: comma/semicolon separated
    if ',' in arg or ';' in arg:
        sep = ',' if ',' in arg else ';'
        return [int(x.strip()) for x in arg.split(sep) if x.strip()]
    # Single integer?
    if arg.strip().lstrip('-').isdigit():
        return [int(arg.strip())]
    raise ValueError("无法解析第二个参数 Set。请传入文件路径、[x,y,z] 或以逗号分隔的整数。")


def next_prime_at_least(n: int) -> int:
    if n <= 2:
        return 2
    if n % 2 == 0:
        n += 1
    while True:
        if is_probable_prime(n):
            return n
        n += 2


def is_probable_prime(n: int) -> bool:
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    # Miller-Rabin
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # Bases sufficient for 64-bit
    for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        skip_to_next_n = False
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                skip_to_next_n = True
                break
        if skip_to_next_n:
            continue
        return False
    return True


def choose_modulus_for_product(product_size: int, target_list_size: int) -> int:
    """Choose a modulus so that expected matched pairs ~ product_size / M ≈ target_list_size.

    For very small products, we disable the modular filter (M=1) to avoid losing rare solutions.
    """
    if product_size <= 0:
        return 1
    # If product is small, do not filter
    if product_size <= max(4, target_list_size // 2):
        return 1
    m_est = max(2, product_size // max(1, target_list_size))
    # keep modulus within reasonable bounds but allow small values
    m_est = min(m_est, 10**8 + 7)
    if m_est <= 3:
        return 1
    return next_prime_at_least(m_est)


def generate_combinations_sums(values: List[int], indices: List[int], k: int) -> List[Entry]:
    if k < 0 or k > len(indices):
        return []
    result: List[Entry] = []
    # Precompute masks for indices for speed
    idx_to_mask: Dict[int, int] = {idx: (1 << idx) for idx in indices}
    for combo_local_pos in combinations(range(len(indices)), k):
        s = 0
        mask = 0
        for pos in combo_local_pos:
            idx = indices[pos]
            s += values[idx]
            mask |= idx_to_mask[idx]
        result.append(Entry(sum_int=s, sum_mod=0, mask=mask))
    return result


def assign_mod(entries: List[Entry], modulus: int) -> List[Entry]:
    if modulus <= 1:
        return [Entry(e.sum_int, 0, e.mask) for e in entries]
    return [Entry(e.sum_int, e.sum_int % modulus, e.mask) for e in entries]


def merge_no_mod(list_a: List[Entry], list_b: List[Entry]) -> List[Entry]:
    # Full Cartesian merge, no modulus filtering
    out: List[Entry] = []
    for ea in list_a:
        sa = ea.sum_int
        ma = ea.mask
        for eb in list_b:
            sb = eb.sum_int
            mb = eb.mask
            out.append(Entry(sa + sb, 0, ma | mb))
    return out


def merge_mod(list_a: List[Entry], list_b: List[Entry], modulus: int, target_residue: int) -> List[Entry]:
    if modulus <= 1:
        # Degenerate: behave like integer target on residue
        # Keep pairs whose (sa + sb) % 1 == target_residue % 1, i.e., all
        return merge_no_mod(list_a, list_b)
    bucket: Dict[int, List[Entry]] = {}
    for eb in list_b:
        bucket.setdefault(eb.sum_mod, []).append(eb)
    out: List[Entry] = []
    tr = target_residue % modulus
    for ea in list_a:
        need = (tr - ea.sum_mod) % modulus
        for eb in bucket.get(need, []):
            out.append(Entry(ea.sum_int + eb.sum_int, (ea.sum_mod + eb.sum_mod) % modulus, ea.mask | eb.mask))
    return out


def merge_mod_by_value(list_a: List[Entry], list_b: List[Entry], modulus: int, target_residue: int) -> List[Entry]:
    """Merge two lists on x.sum_int + y.sum_int ≡ target_residue (mod modulus).

    This computes residues on the fly from sum_int to allow using a different modulus
    than the one stored in sum_mod.
    """
    if modulus <= 1:
        return merge_no_mod(list_a, list_b)
    bucket: Dict[int, List[Entry]] = {}
    for eb in list_b:
        mb = eb.sum_int % modulus
        bucket.setdefault(mb, []).append(eb)
    out: List[Entry] = []
    tr = target_residue % modulus
    for ea in list_a:
        need = (tr - (ea.sum_int % modulus)) % modulus
        for eb in bucket.get(need, []):
            out.append(Entry(ea.sum_int + eb.sum_int, (ea.sum_int + eb.sum_int) % modulus, ea.mask | eb.mask))
    return out


def generate_posneg_entries(values: List[int], indices: List[int], pos_k: int, neg_k: int, sample_cap: Optional[int] = None, rng: Optional[random.Random] = None, deadline: Optional[float] = None) -> List[EntryPN]:
    if pos_k < 0 or neg_k < 0 or pos_k + neg_k > len(indices):
        return []
    pos_combos = list(combinations(indices, pos_k))
    # Optional down-sampling on pos combos to control growth
    if sample_cap is not None and len(pos_combos) > sample_cap:
        assert rng is not None
        pos_combos = rng.sample(pos_combos, sample_cap)
    out: List[EntryPN] = []
    idx_set = set(indices)
    for pos_tuple in pos_combos:
        if deadline is not None and time.time() > deadline:
            break
        if neg_k == 0:
            s = sum(values[i] for i in pos_tuple)
            pos_mask = 0
            for i in pos_tuple:
                pos_mask |= 1 << i
            out.append(EntryPN(s, 0, pos_mask, 0, pos_k, 0))
        else:
            remaining = list(idx_set.difference(pos_tuple))
            neg_iter = combinations(remaining, neg_k)
            # Optional sampling over combined (pos,neg) pairs
            if sample_cap is not None:
                # Convert iterator to limited list by sampling a bounded number of negatives
                neg_list = list(neg_iter)
                if len(neg_list) > sample_cap:
                    neg_list = rng.sample(neg_list, sample_cap)
            else:
                neg_list = list(neg_iter)
            for neg_tuple in neg_list:
                if deadline is not None and time.time() > deadline:
                    break
                s = sum(values[i] for i in pos_tuple) - sum(values[i] for i in neg_tuple)
                pos_mask = 0
                neg_mask = 0
                for i in pos_tuple:
                    pos_mask |= 1 << i
                for i in neg_tuple:
                    neg_mask |= 1 << i
                out.append(EntryPN(s, 0, pos_mask, neg_mask, pos_k, neg_k))
    return out


def assign_mod_pn(entries: List[EntryPN], modulus: int) -> List[EntryPN]:
    if modulus <= 1:
        return [EntryPN(e.sum_int, 0, e.pos_mask, e.neg_mask, e.pos_count, e.neg_count) for e in entries]
    return [EntryPN(e.sum_int, e.sum_int % modulus, e.pos_mask, e.neg_mask, e.pos_count, e.neg_count) for e in entries]


def merge_mod_pn(list_a: List[EntryPN], list_b: List[EntryPN], modulus: int, target_residue: int) -> List[EntryPN]:
    """Merge with modular filter and cancellation-aware consistency:
    - Disallow +1/+1 or -1/-1 overlap (avoid 2 or -2)
    - Allow +1/-1 overlap (cancellation)
    - Recompute pos/neg counts after cancellation
    """
    def make_entry(ea: EntryPN, eb: EntryPN, sum_mod_val: int) -> EntryPN:
        # Disallow same-sign overlaps
        if (ea.pos_mask & eb.pos_mask) != 0:
            return None  # type: ignore
        if (ea.neg_mask & eb.neg_mask) != 0:
            return None  # type: ignore
        pos_union = ea.pos_mask | eb.pos_mask
        neg_union = ea.neg_mask | eb.neg_mask
        # Cancellation-aware counts
        pos_only = pos_union & ~neg_union
        neg_only = neg_union & ~pos_union
        new_pos_count = pos_only.bit_count()
        new_neg_count = neg_only.bit_count()
        return EntryPN(
            ea.sum_int + eb.sum_int,
            sum_mod_val,
            pos_union,
            neg_union,
            new_pos_count,
            new_neg_count,
        )

    out: List[EntryPN] = []
    if modulus <= 1:
        for ea in list_a:
            for eb in list_b:
                ent = make_entry(ea, eb, 0)
                if ent is not None:
                    out.append(ent)
        return out
    bucket: Dict[int, List[EntryPN]] = {}
    for eb in list_b:
        bucket.setdefault(eb.sum_int % modulus, []).append(eb)
    tr = target_residue % modulus
    for ea in list_a:
        need = (tr - (ea.sum_int % modulus)) % modulus
        for eb in bucket.get(need, []):
            ent = make_entry(ea, eb, (ea.sum_int + eb.sum_int) % modulus)
            if ent is not None:
                out.append(ent)
    return out


def find_match_integer_sum_pn(list_y: List[EntryPN], list_z: List[EntryPN], target_sum: int, weight: int) -> Optional[List[int]]:
    # Map sum_y -> entries for z-side iteration to be small or vice versa
    # Choose smaller list to hash for memory
    if len(list_y) > len(list_z):
        list_y, list_z = list_z, list_y
        swapped = True
    else:
        swapped = False
    by_sum: Dict[int, List[EntryPN]] = {}
    for e in list_y:
        by_sum.setdefault(e.sum_int, []).append(e)
    for ez in list_z:
        want = target_sum - ez.sum_int
        if want not in by_sum:
            continue
        for ey in by_sum[want]:
            py, ny = (ey.pos_mask, ey.neg_mask) if not swapped else (ez.pos_mask, ez.neg_mask)
            pz, nz = (ez.pos_mask, ez.neg_mask) if not swapped else (ey.pos_mask, ey.neg_mask)
            # Forbid +1+1 or -1-1 on same index
            if (py & pz) != 0:
                continue
            if (ny & nz) != 0:
                continue
            # Net negative must be zero after cancellation
            pos_union = py | pz
            neg_union = ny | nz
            net_neg = neg_union & ~pos_union
            if net_neg != 0:
                continue
            # Net positive are positions with only +1 and no -1 anywhere
            x_mask = pos_union & ~neg_union
            if x_mask.bit_count() != weight:
                continue
            n = max(py.bit_length(), pz.bit_length(), ny.bit_length(), nz.bit_length())
            chosen = [i for i in range(n) if (x_mask >> i) & 1]
            return chosen
    return None


def merge_no_mod_pn(list_a: List[EntryPN], list_b: List[EntryPN]) -> List[EntryPN]:
    out: List[EntryPN] = []
    for ea in list_a:
        for eb in list_b:
            # Disallow same-sign overlaps
            if (ea.pos_mask & eb.pos_mask) != 0:
                continue
            if (ea.neg_mask & eb.neg_mask) != 0:
                continue
            pos_union = ea.pos_mask | eb.pos_mask
            neg_union = ea.neg_mask | eb.neg_mask
            pos_only = pos_union & ~neg_union
            neg_only = neg_union & ~pos_union
            new_pos_count = pos_only.bit_count()
            new_neg_count = neg_only.bit_count()
            out.append(EntryPN(
                ea.sum_int + eb.sum_int,
                0,
                pos_union,
                neg_union,
                new_pos_count,
                new_neg_count,
            ))
    return out


def filter_consistent(entries: List[EntryPN], want_pos: int, want_neg: int) -> List[EntryPN]:
    out: List[EntryPN] = []
    for e in entries:
        # No 2 or -2 components across merges is ensured by construction if we checked masks on the way.
        # Here we only enforce exact counts at this level.
        if e.pos_count == want_pos and e.neg_count == want_neg:
            out.append(e)
    return out


def build_Lomega(values: List[int], A: List[int], B: List[int], N1: int, Nm: int, rng: random.Random, Momega: int, Rw: int, deadline: Optional[float]) -> Optional[List[EntryPN]]:
    """Construct one Lω list over all n indices as in BCJ:
    - Randomly split indices into two halves of n/2
    - Choose pos and neg counts per half (balanced split)
    - Generate half lists and merge modulo Mω chosen to reach desired_size
    """
    # Build one ω-list using the provided fixed halves A and B (shared across all ω lists)
    # Balanced split
    pos_a = N1 // 2
    pos_b = N1 - pos_a
    neg_a = Nm // 2
    neg_b = Nm - neg_a
    # Small random tweak to avoid rigidity
    if Nm % 2 == 1 and rng.random() < 0.5:
        neg_a, neg_b = neg_b, neg_a
    # Generate half lists (no sampling, rely on small counts for n<=64)
    if deadline is not None and time.time() > deadline:
        return None
    LA = generate_posneg_entries(values, A, pos_a, neg_a, sample_cap=None, rng=rng, deadline=deadline)
    if deadline is not None and time.time() > deadline:
        return None
    LB = generate_posneg_entries(values, B, pos_b, neg_b, sample_cap=None, rng=rng, deadline=deadline)
    if not LA or not LB:
        return []
    # Merge modulo global Mω and residue Rw
    if deadline is not None and time.time() > deadline:
        return None
    Lw = merge_mod_pn(assign_mod_pn(LA, Momega), assign_mod_pn(LB, Momega), Momega, Rw)
    return Lw


def find_match_integer_sum(list_y: List[Entry], list_z: List[Entry], target_sum: int) -> Optional[Tuple[int, int, int]]:
    # Map sum_y -> entries
    by_sum: Dict[int, List[Entry]] = {}
    for e in list_y:
        by_sum.setdefault(e.sum_int, []).append(e)
    for ez in list_z:
        want = target_sum - ez.sum_int
        for ey in by_sum.get(want, []):
            if (ey.mask & ez.mask) == 0:
                return ey.sum_int, ez.sum_int, ey.mask | ez.mask
    return None


def popcount(x: int) -> int:
    return x.bit_count()


def try_once(values: List[int], target_sum: int, weight: int, rng: random.Random, target_list_cap: int = 120_000, deadline: Optional[float] = None) -> Optional[List[int]]:
    """Generic BCJ three-level ω/κ/ν with {-1,0,1} as in the paper (practical variant).

    - Build 8 ω lists with sizes roughly governed by Nω(1), Nω(-1), Nω(0)
    - Merge to 4 κ lists with modular constraint Mω and filter inconsistent (counts)
    - Merge to 2 ν lists with modular constraint Mκ and filter inconsistent
    - Final integer join to target S with cancellation checks producing {0,1}
    """
    n = len(values)
    if weight < 0 or weight > n:
        return None
    # Random permutation for load balancing
    perm = list(range(n))
    rng.shuffle(perm)
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[i] = p
    vals = [values[p] for p in perm]

    # BCJ parameters
    alpha = 0.0267
    beta = 0.0168
    gamma = 0.0029
    # Target counts per level from paper
    Nnu1 = int(round((0.25 + alpha) * n))
    NnuM = int(round(alpha * n))
    Nk1 = int(round((0.125 + alpha / 2.0 + beta) * n))
    NkM = int(round((alpha / 2.0 + beta) * n))
    Nw1 = int(round((0.0625 + alpha / 4.0 + beta / 2.0 + gamma) * n))
    NwM = int(round((alpha / 4.0 + beta / 2.0 + gamma) * n))

    # Step 1: build 8 Lω lists using one shared random split (A,B) and global Mω
    desired_Lw = 20000
    # Shared halves
    idx = list(range(n))
    rng.shuffle(idx)
    A = idx[: n // 2]
    B = idx[n // 2 :]
    # Global Mω: choose from target per-list size ~ desired_Lw
    # The product of two half-lists is implicit; we set Mω to shrink to desired_Lw directly
    Momega = next_prime_at_least(max(10007, desired_Lw))
    Lw: List[List[EntryPN]] = []
    for _ in range(8):
        if deadline is not None and time.time() > deadline:
            return None
        Rw = rng.randrange(Momega)
        L = build_Lomega(vals, A, B, Nw1, NwM, rng, Momega, Rw, deadline)
        if L is None:
            return None
        if not L:
            continue
        Lw.append(L)
    if len(Lw) < 8:
        return None
    # Debug sizes at ω
    print(f"[BCJ] Lω sizes: {[len(L) for L in Lw]} Nω(1)={Nw1} Nω(-1)={NwM}")

    # Step 2: merge to four κ lists with modulus and filter counts to Nk1/NkM
    # Global Mκ and coupled Rκ per pair consistent with ω residues (simplified: reuse Mκ across pairs)
    Mkappa = choose_modulus_for_product(max(1, sum(len(L) for L in Lw)) ** 2, 50000)
    if Mkappa <= 1:
        Mkappa = next_prime_at_least(10007)

    def merge_two_lists_to_kappa(A: List[EntryPN], B: List[EntryPN]) -> List[EntryPN]:
        Rk = rng.randrange(Mkappa)
        Ktmp = merge_mod_pn(assign_mod_pn(A, Mkappa), assign_mod_pn(B, Mkappa), Mkappa, Rk)
        # Filter with tolerance ±1 for small n
        K = [e for e in Ktmp if abs(e.pos_count - Nk1) <= 1 and abs(e.neg_count - NkM) <= 1]
        # Cap
        if len(K) > target_list_cap:
            K = rng.sample(K, target_list_cap)
        return K

    K1 = merge_two_lists_to_kappa(Lw[0], Lw[1])
    K2 = merge_two_lists_to_kappa(Lw[2], Lw[3])
    K3 = merge_two_lists_to_kappa(Lw[4], Lw[5])
    K4 = merge_two_lists_to_kappa(Lw[6], Lw[7])
    if not K1 or not K2 or not K3 or not K4:
        return None
    print(f"[BCJ] Kappa sizes: {[len(K) for K in [K1,K2,K3,K4]]} Nk(1)={Nk1} Nk(-1)={NkM}")

    # Step 3: choose ONE Mν and enforce target S residues across left/right
    prod_left = max(1, len(K1)) * max(1, len(K2))
    prod_right = max(1, len(K3)) * max(1, len(K4))
    prod_nu = min(prod_left, prod_right)
    Mn = choose_modulus_for_product(prod_nu, 50000)
    if Mn <= 1:
        Mn = next_prime_at_least(10007)
    Rleft = rng.randrange(Mn)
    Rright = (target_sum % Mn - Rleft) % Mn

    def merge_two_kappa_to_nu(A: List[EntryPN], B: List[EntryPN], target_residue: int) -> List[EntryPN]:
        Ntmp = merge_mod_pn(assign_mod_pn(A, Mn), assign_mod_pn(B, Mn), Mn, target_residue)
        Nf = [e for e in Ntmp if abs(e.pos_count - Nnu1) <= 1 and abs(e.neg_count - NnuM) <= 1]
        if len(Nf) > target_list_cap:
            Nf = rng.sample(Nf, target_list_cap)
        return Nf

    Nleft = merge_two_kappa_to_nu(K1, K2, Rleft)
    Nright = merge_two_kappa_to_nu(K3, K4, Rright)
    if not Nleft or not Nright:
        return None
    print(f"[BCJ] Nu sizes: left={len(Nleft)} right={len(Nright)} Nν(1)={Nnu1} Nν(-1)={NnuM}")

    # Final: integer join Nleft + Nright = S with cancellation -> {0,1}
    res = find_match_integer_sum_pn(Nleft, Nright, target_sum, weight)
    if res is None:
        return None
    chosen_original = sorted(inv_perm[i] for i in res)
    return chosen_original


def solve(values: List[int], target_sum: int, weight: int, max_attempts: int = 200, seed: Optional[int] = None, time_budget_s: Optional[float] = 30.0) -> Optional[List[int]]:
    rng = random.Random(seed if seed is not None else int(time.time() * 1000) ^ os.getpid())
    # Try a few quick exact fallbacks for tiny n to improve robustness (kept)
    n = len(values)
    if n <= 22:
        idxs = list(range(n))
        for comb in combinations(idxs, weight):
            s = sum(values[i] for i in comb)
            if s == target_sum:
                return list(comb)
    # Time budget handling
    deadline = None
    if time_budget_s is not None:
        deadline = time.time() + float(time_budget_s)
    for _ in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        res = try_once(values, target_sum, weight, rng, deadline=deadline)
        if res is not None:
            return res
    return None


# ------------------ Specialized block-structure BCJ (6 choose 3 per block) ------------------

def block_combos(values: List[int], base_index: int, block_size: int = 6, k: int = 3) -> List[Entry]:
    """Generate all k-of-block_size combinations for a contiguous block starting at base_index.

    Returns list of Entry with sum and global mask.
    """
    idxs = list(range(base_index, base_index + block_size))
    out: List[Entry] = []
    for comb in combinations(idxs, k):
        s = sum(values[i] for i in comb)
        mask = 0
        for i in comb:
            mask |= 1 << i
        out.append(Entry(s, 0, mask))
    return out


def merge_sorted_by_sum(list_a: List[Entry], list_b: List[Entry]) -> List[Entry]:
    """Cartesian merge A+B without modulus, returning combined list. For small lists only."""
    out: List[Entry] = []
    for ea in list_a:
        sa = ea.sum_int
        ma = ea.mask
        for eb in list_b:
            out.append(Entry(sa + eb.sum_int, 0, ma | eb.mask))
    return out


# def solve_block6_choose3(values: List[int], target_sum: int) -> Optional[List[int]]:
#     n = len(values)
#     if n % 6 != 0:
#         return None
#     num_blocks = n // 6
#     if num_blocks != 8:
#         # generic but tuned for 8 blocks case
#         pass

#     # Build ω-level lists: per block choose-3-of-6
#     Lw: List[List[Entry]] = []
#     for b in range(num_blocks):
#         Lw.append(block_combos(values, b * 6, 6, 3))

#     # κ-level: pairwise merge blocks 0-1, 2-3, 4-5, 6-7
#     Kk: List[List[Entry]] = []
#     for j in range(0, num_blocks, 2):
#         Kk.append(merge_sorted_by_sum(Lw[j], Lw[j + 1]))  # size ≈ 400 each

#     # ν-level: merge pairs of κ-lists -> two big lists (≈160k each)
#     Kn_left = merge_sorted_by_sum(Kk[0], Kk[1])   # blocks 0..3
#     Kn_right = merge_sorted_by_sum(Kk[2], Kk[3])  # blocks 4..7

#     # Final integer collision: Kn_left + Kn_right = target_sum
#     # Build hashmap for right side sums → masks
#     sum_to_masks: Dict[int, List[int]] = {}
#     for e in Kn_right:
#         sum_to_masks.setdefault(e.sum_int, []).append(e.mask)
#     for el in Kn_left:
#         need = target_sum - el.sum_int
#         if need in sum_to_masks:
#             ml = el.mask
#             for mr in sum_to_masks[need]:
#                 # masks are disjoint by construction (different blocks), but we keep check
#                 if (ml & mr) == 0:
#                     mask = ml | mr
#                     # Convert to sorted indices
#                     chosen = [i for i in range(n) if (mask >> i) & 1]
#                     return chosen
#     return None


def main():
    if len(sys.argv) != 4:
        print("用法: python bcj.py <Sum> <Set> <HammingWeight>")
        print("示例: python bcj.py 100 \"[3,34,4,12,5,2]\" 3")
        print("或:    python bcj.py 100 3,34,4,12,5,2 3")
        print("或:    python bcj.py 100 weights.txt 3   # 文件内按空格/换行分隔整数")
        sys.exit(1)

    try:
        target_sum = int(sys.argv[1])
    except Exception:
        print("第一个参数 Sum 必须是整数")
        sys.exit(2)

    try:
        weights = parse_set_arg(sys.argv[2])
    except Exception as e:
        print(str(e))
        sys.exit(3)

    try:
        ham_w = int(sys.argv[3])
    except Exception:
        print("第三个参数 HammingWeight 必须是整数")
        sys.exit(4)

    if ham_w < 0 or ham_w > len(weights):
        print("HammingWeight 超出范围")
        sys.exit(5)

    # Generic BCJ by default (no specialized solver)
    res: Optional[List[int]] = solve(weights, target_sum, ham_w)
    if res is None:
        print("未找到满足约束的解（可增加尝试次数或更换随机种子）。")
        sys.exit(6)

    # 输出结果
    chosen = res
    chosen_vals = [weights[i] for i in chosen]
    s = sum(chosen_vals)
    x_vec = [1 if i in set(chosen) else 0 for i in range(len(weights))]
    print("Found solution:")
    print("\tIndices (0-based):", chosen)
    print("\tCorresponding Elements:", chosen_vals)
    # print("verify: sum=", s, " -- target ", target_sum)
    print(f"\tSum: {s} -- target {target_sum} -- {'OK' if s == target_sum else 'MISMATCH'}")
    print("\tHammingWeight:", len(chosen))
    print("Solution x ")
    print("\tVector:", x_vec)

    e_str = ''.join(str(bit) for bit in x_vec)
    print("\tString:", e_str)
    print("\tSHA-256:", hashlib.sha256(e_str.encode()).hexdigest())
    # to Bytes
    try:
        byte_len = (len(e_str) + 7) // 8
        e_bytes = int(e_str, 2).to_bytes(byte_len, 'big')
        print("\tBytes:", e_bytes)
    except Exception as e:
        print("\tError converting to bytes:", e)


if __name__ == "__main__":
    main()
