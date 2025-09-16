#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solve_knapsack_debug.py — 求解给定的 48×96 低密度背包实例并恢复 p（LSB-first）
使用 LO 构造 + LLL。

运行：
  sage -python solve_knapsack_debug.py
"""
from __future__ import annotations
from typing import List

from sage.all import Matrix, ZZ

# 来自题面 DEBUG 样例（a 长度 48，每个 ~96bit；bag 为整数）

# flag{0xb438b658e260}
a=[44806550944109149068319223837, 53826386082770453251646787179, 45786535336691951552788270949, 56723377661009264786262367297, 42834505068802016142588734867, 59833863366095987873690217359, 61596373738280030693656147081, 51493583327453918642966867897, 43227201525454341087385780267, 49631779730151021762315617501, 71041040226995721551636296529, 62184240581604211474534592821, 53551463689810054718067231691, 56155139075735612297850315511, 59118305839178161392670840981, 49411832803457473527918557179, 60422164246141487709889548559, 50576337719464642694296408921, 68026215025965093228226207031, 45706002821587125154134528617, 44686518753683588099137842181, 41803534803097179377098363423, 67928004101435014362072672713, 44892450066256615633819197781, 49457322680480848634324498849, 40629241028389251269008407721, 40298086347714588217400828261, 70254052856175832442254545433, 42388862258384511558093665569, 45937929917293028061648243431, 60125904395671915673765478461, 40007696083741011497053757539, 57539146968627795201868020157, 51696432796569276746521009553, 66969708209088246417081922049, 76675638517922649468361216039, 78026625403562059965348558229, 52183336535985557903040200977, 65400325143586860937422518867, 40295547637073556912525830611, 58567130238688373671234050877, 41307718893812824325689892263, 49662247286611096497649810259, 58347430392474250167570212051, 49982844030796181864167434793, 58713957787038961536493678751, 50088398284179597731997071753, 62406119513446926107864687737]
bag=1130980404940489839109577228429

n = len(a)

# 构造 LO 基： [ I_n | a ] ; [ 0 | -C ]
A = Matrix(ZZ, n + 1, n + 1)
for i in range(n):
    A[i, i] = 1
    A[i, n] = int(a[i])
A[n, n] = -int(bag)

# 先做一次 LLL
res = A.LLL(delta=0.99)

solution_bits: List[int] = []
found = False
for i in range(n + 1):
    row = [int(x) for x in res.row(i).list()]
    if row[-1] != 0:
        continue
    # 允许整体取反
    for cand in (row, [-x for x in row]):
        head = cand[:n]
        if all(abs(x) in (0, 1) for x in head):
            bits = [abs(x) for x in head]
            # 校验
            if sum(ai * bi for ai, bi in zip(a, bits)) == bag:
                solution_bits = bits
                found = True
                break
    if found:
        break

if not found:
    # 尝试 CJLOSS（整数化，scale=2）：
    print("[*] LO 未找到解，尝试 CJLOSS 方法（scale=2）...")
    # [ 2I | a ] ; [ 1..1 | C ]，寻找 (2e-1, 0)
    B = Matrix(ZZ, n + 1, n + 1)
    for i in range(n):
        B[i, i] = 2
        B[i, n] = int(a[i])
    for j in range(n):
        B[n, j] = 1
    B[n, n] = int(bag)
    B = B.LLL(delta=0.99)
    for i in range(n + 1):
        row = [int(x) for x in B.row(i).list()]
        if row[-1] != 0:
            continue
        for cand in (row, [-x for x in row]):
            head = cand[:n]
            # 期待为 ±1
            if all(abs(x) == 1 for x in head):
                # bits = (s+1)//2
                bits = [(s + 1) // 2 for s in head]
                if sum(ai * bi for ai, bi in zip(a, bits)) == bag:
                    solution_bits = bits
                    found = True
                    break
        if found:
            break

if not found:
    print("[!] 未找到解；可尝试增大 N 或使用 BKZ。")
else:
    # LSB-first 重构 p：p = sum(bits[i] << i)
    p_val = 0
    for i, b in enumerate(solution_bits):
        if b:
            p_val |= (1 << i)
    print("[+] 找到解！")
    print(f"bits (lsb->msb): {''.join(str(b) for b in solution_bits)}")
    print(f"p (int)        : {p_val}")
    print(f"p (hex)        : {hex(p_val)}")
    # 再次校验 bag
    assert sum(ai * bi for ai, bi in zip(a, solution_bits)) == bag
    print("[+] 校验通过：sum(a_i * bit_i) == bag")
