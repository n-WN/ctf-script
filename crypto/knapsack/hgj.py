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


def try_once(values: List[int], target_sum: int, weight: int, rng: random.Random, target_list_cap: int = 120_000) -> Optional[List[int]]:
    n = len(values)
    if weight < 0 or weight > n:
        return None
    # Random permutation to help balancing
    perm = list(range(n))
    rng.shuffle(perm)
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    vals = [values[p] for p in perm]

    # Split into halves then quarters: [A1, A2, B1, B2]
    half = n // 2
    quarter = n // 4
    A1 = list(range(0, quarter))
    A2 = list(range(quarter, half))
    B1 = list(range(half, half + quarter))
    B2 = list(range(half + quarter, n))

    # Distribute weight across y/z and quarters randomly but feasibly
    # y has wy, z has wz, with wy + wz = weight
    wy = weight // 2
    wz = weight - wy

    def sample_distribution(caps: List[int], total: int) -> List[int]:
        # Sample a bounded composition x_i with 0 <= x_i <= caps[i], sum x_i = total
        # Greedy randomized within feasible bounds
        out: List[int] = []
        remaining = total
        for i, cap in enumerate(caps):
            max_future = sum(caps[i+1:]) if i + 1 < len(caps) else 0
            min_i = max(0, remaining - max_future)
            max_i = min(cap, remaining)
            if max_i < min_i:
                return []  # infeasible
            xi = rng.randint(min_i, max_i)
            out.append(xi)
            remaining -= xi
        if remaining != 0:
            return []
        return out

    caps = [len(A1), len(A2), len(B1), len(B2)]
    # Retry a few times to get feasible draws
    for _ in range(10):
        y_quarters = sample_distribution(caps, wy)
        z_quarters = sample_distribution(caps, wz)
        if y_quarters and z_quarters:
            break
    if not y_quarters or not z_quarters:
        return None

    # Ensure feasibility
    groups = [A1, A2, B1, B2]
    for k, grp in zip(y_quarters, groups):
        if k < 0 or k > len(grp):
            return None
    for k, grp in zip(z_quarters, groups):
        if k < 0 or k > len(grp):
            return None

    # Build quarter lists for y
    YA1 = generate_combinations_sums(vals, A1, y_quarters[0])
    YA2 = generate_combinations_sums(vals, A2, y_quarters[1])
    YB1 = generate_combinations_sums(vals, B1, y_quarters[2])
    YB2 = generate_combinations_sums(vals, B2, y_quarters[3])

    # Build quarter lists for z
    ZA1 = generate_combinations_sums(vals, A1, z_quarters[0])
    ZA2 = generate_combinations_sums(vals, A2, z_quarters[1])
    ZB1 = generate_combinations_sums(vals, B1, z_quarters[2])
    ZB2 = generate_combinations_sums(vals, B2, z_quarters[3])

    # Choose bottom modulus M1 to prune quarter merges
    prod_y_q1 = max(1, len(YA1)) * max(1, len(YA2))
    prod_y_q2 = max(1, len(YB1)) * max(1, len(YB2))
    prod_z_q1 = max(1, len(ZA1)) * max(1, len(ZA2))
    prod_z_q2 = max(1, len(ZB1)) * max(1, len(ZB2))
    prod_q_min = min(prod_y_q1, prod_y_q2, prod_z_q1, prod_z_q2)
    # Aim for around 50k pairs at this stage
    M1 = choose_modulus_for_product(prod_q_min, 50_000)
    if M1 == 1:
        # fallback to small prime to get some pruning
        M1 = next_prime_at_least(10007)

    # Assign residues for M1
    YA1_m1 = assign_mod(YA1, M1)
    YA2_m1 = assign_mod(YA2, M1)
    YB1_m1 = assign_mod(YB1, M1)
    YB2_m1 = assign_mod(YB2, M1)
    ZA1_m1 = assign_mod(ZA1, M1)
    ZA2_m1 = assign_mod(ZA2, M1)
    ZB1_m1 = assign_mod(ZB1, M1)
    ZB2_m1 = assign_mod(ZB2, M1)

    # Random target residues for bottom merges (independent of top constraint)
    RA1 = rng.randrange(M1)
    RB1 = rng.randrange(M1)
    RZA1 = rng.randrange(M1)
    RZB1 = rng.randrange(M1)

    # Merge quarters to halves with modulus M1 to prune pairs
    Y_A = merge_mod(YA1_m1, YA2_m1, M1, RA1)
    Y_B = merge_mod(YB1_m1, YB2_m1, M1, RB1)
    Z_A = merge_mod(ZA1_m1, ZA2_m1, M1, RZA1)
    Z_B = merge_mod(ZB1_m1, ZB2_m1, M1, RZB1)

    # Choose top modulus M2 for half merges (independent from M1)
    prod_y = max(1, len(Y_A)) * max(1, len(Y_B))
    prod_z = max(1, len(Z_A)) * max(1, len(Z_B))
    target_size = target_list_cap
    M2 = choose_modulus_for_product(min(prod_y, prod_z), target_size)
    if M2 == 1:
        # ensure coprime and non-trivial
        M2 = next_prime_at_least(1000003)
        if M2 == M1:
            M2 = next_prime_at_least(M1 + 2)
    R2 = rng.randrange(M2)

    # Build y and z lists with modular constraints under M2
    LY = merge_mod_by_value(Y_A, Y_B, M2, R2)
    LZ = merge_mod_by_value(Z_A, Z_B, M2, (target_sum % M2 - R2) % M2)

    # If lists are too big, randomly subsample to cap memory
    def maybe_sample(lst: List[Entry], cap: int) -> List[Entry]:
        if len(lst) <= cap:
            return lst
        return rng.sample(lst, cap)

    cap_each = target_list_cap
    LY = maybe_sample(LY, cap_each)
    LZ = maybe_sample(LZ, cap_each)

    # Try to find an integer-sum match with disjoint masks
    match = find_match_integer_sum(LY, LZ, target_sum)
    if match is None:
        return None
    _, _, mask = match

    # Check weight
    if popcount(mask) != weight:
        return None

    # Convert mask back to original indices
    chosen_perm_indices: List[int] = [i for i in range(n) if (mask >> i) & 1]
    chosen_original_indices: List[int] = sorted(perm[i] for i in chosen_perm_indices)
    return chosen_original_indices


def solve(values: List[int], target_sum: int, weight: int, max_attempts: int = 200, seed: Optional[int] = None) -> Optional[List[int]]:
    rng = random.Random(seed if seed is not None else int(time.time() * 1000) ^ os.getpid())
    # Try a few quick exact fallbacks for tiny n to improve robustness
    n = len(values)
    if n <= 22:  # safe bound
        # Enumerate all masks of given weight and check
        idxs = list(range(n))
        for comb in combinations(idxs, weight):
            s = sum(values[i] for i in comb)
            if s == target_sum:
                return list(comb)
    for _ in range(max_attempts):
        res = try_once(values, target_sum, weight, rng)
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


def solve_block6_choose3(values: List[int], target_sum: int) -> Optional[List[int]]:
    n = len(values)
    if n % 6 != 0:
        return None
    num_blocks = n // 6
    if num_blocks != 8:
        # generic but tuned for 8 blocks case
        pass

    # Build ω-level lists: per block choose-3-of-6
    Lw: List[List[Entry]] = []
    for b in range(num_blocks):
        Lw.append(block_combos(values, b * 6, 6, 3))

    # κ-level: pairwise merge blocks 0-1, 2-3, 4-5, 6-7
    Kk: List[List[Entry]] = []
    for j in range(0, num_blocks, 2):
        Kk.append(merge_sorted_by_sum(Lw[j], Lw[j + 1]))  # size ≈ 400 each

    # ν-level: merge pairs of κ-lists -> two big lists (≈160k each)
    Kn_left = merge_sorted_by_sum(Kk[0], Kk[1])   # blocks 0..3
    Kn_right = merge_sorted_by_sum(Kk[2], Kk[3])  # blocks 4..7

    # Final integer collision: Kn_left + Kn_right = target_sum
    # Build hashmap for right side sums → masks
    sum_to_masks: Dict[int, List[int]] = {}
    for e in Kn_right:
        sum_to_masks.setdefault(e.sum_int, []).append(e.mask)
    for el in Kn_left:
        need = target_sum - el.sum_int
        if need in sum_to_masks:
            ml = el.mask
            for mr in sum_to_masks[need]:
                # masks are disjoint by construction (different blocks), but we keep check
                if (ml & mr) == 0:
                    mask = ml | mr
                    # Convert to sorted indices
                    chosen = [i for i in range(n) if (mask >> i) & 1]
                    return chosen
    return None


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

    # First, if the structure matches 8 blocks of 6 with weight=24, try specialized block solver
    res: Optional[List[int]] = None
    if len(weights) == 48 and ham_w == 24:
        res = solve_block6_choose3(weights, target_sum)
    if res is None:
        res = solve(weights, target_sum, ham_w)
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
