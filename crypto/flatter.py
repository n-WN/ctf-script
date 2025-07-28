# https://tangcuxiaojikuai.xyz/post/ec560d26.html
import os
from re import findall
from subprocess import check_output

def flatter(M):
    # compile https://github.com/keeganryan/flatter and put it in $PATH
    z = "[[" + "]\n[".join(" ".join(map(str, row)) for row in M) + "]]"
    env = os.environ.copy
    env［'OMP_NUM_THREADS'］ = '6' # MacBook上控制线程数量, 避免调度到小核上, 几个大核就填几
    ret = check_output(["flatter"], input=z.encode())
    return matrix(M.nrows(), M.ncols(), map(int, findall(b"-?\\d+", ret)))
