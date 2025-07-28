from sage.all import PolynomialRing, Zmod, inverse_mod
from Crypto.Util.number import *
from tqdm import trange
from multiprocessing import Process, Queue
import multiprocessing
import time

e = 149
d0 = 6001958312144157007304943931113872134090201010357773442954181100786589106572169
n = 88436063749749362190546240596734626745594171540325086418270903156390958817492063940459108934841028734921718351342152598224670602551497995639921650979296052943953491639892805985538785565357751736799561653032725751622522198746331856539251721033316195306373318196300612386897339425222615697620795751869751705629
c = 1206332017436789083799133504302957634035030644841294494992108068042941783794804420630684301945366528832108224264145563741764232409333108261614056154508904583078897178425071831580459193200987943565099780857889864631160747321901113496943710282498262354984634755751858251005566753245882185271726628553508627299

def run_small_roots_worker(n, plow, result_queue):
    """在独立进程中运行small_roots"""
    try:
        R = PolynomialRing(Zmod(n), 'x')
        x = R.gen()
        f1 = x * 2^265 + plow
        f1 = f1.monic()
        res = f1.small_roots(X=2^247, beta=0.5, epsilon=0.01)
        result_queue.put(res)
    except Exception as e:
        result_queue.put(None)

def run_small_roots_with_timeout(n, plow, timeout=30):
    """使用multiprocessing实现真正的超时控制"""
    result_queue = Queue()
    
    # 创建子进程
    process = Process(target=run_small_roots_worker, args=(n, plow, result_queue))
    process.start()

    # 等待指定时间
    process.join(timeout)
    
    if process.is_alive():
        # 超时，强制终止进程
        process.terminate()
        process.join()
        print(f"small_roots运行超时({timeout}秒)，已终止进程")
        return None
    
    # 获取结果
    try:
        return result_queue.get_nowait()
    except:
        return None

# 主程序
if __name__ == '__main__':
    found = False
    for k in trange(1, e):
        if found:
            break

        R.<p> = PolynomialRing(Zmod(2^265))
        temp = e * d0 * p
        f = (k * n * p - k * p^2 - k * n + k * p + p) - temp
        # roots = solve_mod([f], 2^265)

        roots_ = f.roots(multiplicities=False)

        # print(type(roots_))

        # print(roots_[0])

        print(f"尝试 k={k}, roots={roots_}")
        
        if roots_ != []:
            for root in roots_:
                plow = int(root)
                # print(f"尝试 k={k}, plow={plow}")
                
                try:
                    res = run_small_roots_with_timeout(n, plow, timeout=30)
                    
                    if res is not None and res != []:
                        print(f"\n找到结果: {res}")
                        # [188210689227294472160085325314952069542671020803828390144430392548173787275]
                        
                        p = int(res[0]) * 2^265 + plow
                        print(f"p = {p}")
                        # p = 11158174168280917736979570452068827611755694573672250873587467083259280584739528118050085070475912733864211083865201596017044398008278425498714490994488939
                        
                        q = n // p
                        d = inverse_mod(e, (p-1)*(q-1))
                        m = pow(c, d, n)
                        flag = long_to_bytes(int(m))
                        print(f"解密结果: {flag}")
                        # flag{827ccb0eea8a706c4c34a16891f84e7b}
                        found = True
                        break
                        
                except Exception as error:
                    print(f"k={k}时发生错误: {error}")
                    continue

    print("搜索完成")
