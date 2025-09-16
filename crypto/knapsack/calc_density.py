#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_density.py — 计算给定公钥 pubKey 的背包密度与相关统计

用法：
  python3 calc_density.py

期望当前目录存在 pk.py，且定义了 pubKey 列表（整数）。
输出：
  - n（维度）
  - max(a_i) 的 bit 长度 m
  - 背包密度 d = n / m（Lagarias–Odlyzko 定义）
  - 其他统计：最小/最大/平均 bit 长度
  - 基于阈值的粗略判定：
      d < 0.6463  → 低密度区（LO 理论阈值）
      d < 0.9408  → 中密度区（CJLOSS 理论阈值）
      其它        → 高密度区（更难）
"""
from __future__ import annotations
import os
import sys
from statistics import mean
from typing import List, Tuple


def read_pubkey(pk_path: str = "pk.py") -> List[int]:
    if not os.path.exists(pk_path):
        print(f"[!] 未找到 {pk_path}", file=sys.stderr)
        sys.exit(1)
    ns: dict = {}
    with open(pk_path, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, pk_path, "exec"), ns)
    if "pubKey" not in ns:
        print("[!] pk.py 中未定义 pubKey", file=sys.stderr)
        sys.exit(1)
    pubKey_raw = ns["pubKey"]
    pubKey: List[int] = []
    for x in pubKey_raw:
        if isinstance(x, int):
            pubKey.append(x)
        else:
            s = str(x).strip().rstrip("L").strip()
            pubKey.append(int(s))
    return pubKey


def compute_density(pubKey: List[int]) -> Tuple[float, int, int, float, int, int]:
    n = len(pubKey)
    if n == 0:
        raise ValueError("pubKey 为空")
    abs_vals = [abs(int(a)) for a in pubKey]
    max_a = max(abs_vals)
    if max_a == 0:
        # 非典型情况：所有元素为 0
        m = 1
    else:
        m = max_a.bit_length()
    # Lagarias–Odlyzko 密度
    d = n / m if m > 0 else float("inf")

    bit_lengths = [v.bit_length() if v > 0 else 1 for v in abs_vals]
    min_bits = min(bit_lengths)
    max_bits = max(bit_lengths)
    avg_bits = float(mean(bit_lengths))
    return d, n, m, avg_bits, min_bits, max_bits


def classify_density(d: float) -> str:
    if d < 0.6463:
        return "低密度：LO 理论阈值内；多数随机实例可在理论上被解"
    if d < 0.9408:
        return "中密度：CJLOSS 理论阈值内；需更强规约（较大 BKZ 块）"
    return "高密度：超出已知理论阈值；通常更困难"


def main():
    pubKey = read_pubkey()
    d, n, m, avg_bits, min_bits, max_bits = compute_density(pubKey)

    print("=== Knapsack Density Report ===")
    print(f"n (dimension)     : {n}")
    print(f"max(a_i) bitlen m : {m}")
    print(f"density d = n/m   : {d:.6f}")
    print(f"bitlen stats      : min={min_bits}, avg={avg_bits:.2f}, max={max_bits}")
    print(f"classification    : {classify_density(d)}")


if __name__ == "__main__":
    main()
