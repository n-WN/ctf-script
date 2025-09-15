import time
import secrets
import logging
from math import gcd
from sage.all import GF, Integer, Zp

# Logger 配置
logger = logging.getLogger("amm")
if not logger.handlers:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

# 可选：gmpy2 加速路径
try:
    import gmpy2
    _GMP = True
    mpz = gmpy2.mpz
    def powmod(a, e, m):
        return int(gmpy2.powmod(mpz(a), mpz(e), mpz(m)))
    def invmod(a, m):
        return int(gmpy2.invert(mpz(a), mpz(m)))
except Exception:
    _GMP = False
    def powmod(a, e, m):
        return pow(a, e, m)
    def invmod(a, m):
        return pow(a, -1, m)

# 尝试使用 Crypto 的 long_to_bytes，不存在则退回到内置实现
try:
    from Crypto.Util.number import long_to_bytes as l2b
except Exception:
    def l2b(n: int) -> bytes:
        if n == 0:
            return b"\x00"
        l = (n.bit_length() + 7) // 8
        return n.to_bytes(l, 'big')

# --- Problem Parameters ---
n = 6249734963373034215610144758924910630356277447014258270888329547267471837899275103421406467763122499270790512099702898939814547982931674247240623063334781529511973585977522269522704997379194673181703247780179146749499072297334876619475914747479522310651303344623434565831770309615574478274456549054332451773452773119453059618433160299319070430295124113199473337940505806777950838270849
e = 641747
c = 730024611795626517480532940587152891926416120514706825368440230330259913837764632826884065065554839415540061752397144140563698277864414584568812699048873820551131185796851863064509294123861487954267708318027370912496252338232193619491860340395824180108335802813022066531232025997349683725357024257420090981323217296019482516072036780365510855555146547481407283231721904830868033930943

# n = q^k，自动计算 k_max = v_q(n)
q = 91027438112295439314606669837102361953591324472804851543344131406676387779969

def v_p(nv: int, p: int) -> int:
    k = 0
    while nv % p == 0:
        nv //= p
        k += 1
    return k

k_max = v_p(n, q)

def find_field_root_and_unity(c_mod_p, p, e):
    """
    在模素数 p 下：找到一个 e 次根 one_root 和一个 m 阶单位根 zeta（m=gcd(e,p-1)）。
    返回 (int(one_root), int(zeta), m)。
    """
    Fp = GF(p)
    c_fp = Fp(c_mod_p)
    one_root = c_fp.nth_root(e)  # 若不存在会抛 ValueError
    m_val = gcd(e, p - 1)
    if m_val == 1:
        return int(one_root), 1, m_val
    g = Fp.multiplicative_generator()
    zeta = g**((p - 1) // m_val)
    return int(one_root), int(zeta), m_val

# 兼容旧名
find_one_root_and_unity_mod_p = find_field_root_and_unity

def hensel_lift_root_linear(c, e, p, k_max, x0):
    """
    线性精度版本：将一个 e 次根 x0（模 p）提升到模 p^k_max 的根。
    返回 x（模 p^k_max）。
    """
    # 预计算 p 的幂以及 c 在这些模下的值
    p_pows = [1]
    for _ in range(k_max):
        p_pows.append(p_pows[-1] * p)
    c_mods = [c % p_pows[k] for k in range(k_max + 1)]

    # 预计算导数逆元（模 p）
    f_prime_mod_p = (e * pow(x0 % p, e - 1, p)) % p
    inv_f_prime = pow(f_prime_mod_p, -1, p)

    x = int(x0 % p)
    for k in range(2, k_max + 1):
        p_k = p_pows[k]
        p_km1 = p_pows[k - 1]
        f_mod = (powmod(x, e, p_k) - c_mods[k]) % p_k
        t = (-(f_mod // p_km1) * inv_f_prime) % p
        x = (x + t * p_km1) % p_k
    return x


def hensel_lift_root_newton(c, e, p, k_max, x0):
    """
    倍增精度（Newton）版本：从模 p 的简单根 x0 开始，
    每次将精度从 p^t 提升到 p^{min(2t, k_max)}，直到达到 p^k_max。
    返回 x（模 p^k_max）。
    """
    # 当前精度 t=1（模 p）
    t = 1
    x = int(x0 % p)
    while t < k_max:
        t2 = min(2 * t, k_max)
        mod_t2 = pow(p, t2)
        # f(x) = x^e - c (mod p^{t2})
        f_val = (powmod(x, e, mod_t2) - (c % mod_t2)) % mod_t2
        # f'(x) = e * x^{e-1} (mod p^{t2})
        fprime = (e * powmod(x, e - 1, mod_t2)) % mod_t2
        inv_fprime = invmod(fprime, mod_t2)
        x = (x - (f_val * inv_fprime) % mod_t2) % mod_t2
        t = t2
    return x


def hensel_lift_unity_linear(m, p, k_max, zeta0):
    """
    线性精度版本：将一个 m 阶单位根 zeta0（模 p）提升到模 p^k_max 的单位根。
    返回 zeta（模 p^k_max）。
    """
    # 预计算 p 的幂
    p_pows = [1]
    for _ in range(k_max):
        p_pows.append(p_pows[-1] * p)

    # g(x) = x^m - 1, g'(x) = m x^{m-1}，模 p 的逆元只需计算一次
    g_prime_mod_p = (m * pow(zeta0 % p, m - 1, p)) % p
    inv_g_prime = pow(g_prime_mod_p, -1, p)

    z = int(zeta0 % p)
    for k in range(2, k_max + 1):
        p_k = p_pows[k]
        p_km1 = p_pows[k - 1]
        g_mod = (powmod(z, m, p_k) - 1) % p_k
        t = (-(g_mod // p_km1) * inv_g_prime) % p
        z = (z + t * p_km1) % p_k
    return z


def hensel_lift_unity_newton(m, p, k_max, zeta0):
    """
    倍增精度（Newton）版本：单位根的提升，每次将精度从 p^t 提升到 p^{min(2t, k_max)}。
    返回 zeta（模 p^k_max）。
    """
    t = 1
    z = int(zeta0 % p)
    while t < k_max:
        t2 = min(2 * t, k_max)
        mod_t2 = pow(p, t2)
        # g(x) = x^m - 1, g'(x) = m * x^{m-1}
        g_val = (powmod(z, m, mod_t2) - 1) % mod_t2
        gprime = (m * powmod(z, m - 1, mod_t2)) % mod_t2
        inv_gprime = invmod(gprime, mod_t2)
        z = (z - (g_val * inv_gprime) % mod_t2) % mod_t2
        t = t2
    return z


def teichmueller_lift_doubling(a0_mod_p, p, k_max):
    """
    Teichmüller 提升：从模 p 的单位 a0 开始，使用 a <- a^p (mod p^{2t}) 进行精度倍增，直到 p^{k_max}。
    返回提升结果（模 p^k_max）。
    """
    t = 1
    a = int(a0_mod_p % p)
    if a % p == 0:
        raise ValueError("teichmueller_lift_doubling expects a unit mod p")
    while t < k_max:
        t2 = min(2 * t, k_max)
        mod_t2 = pow(p, t2)
        a = powmod(a, p, mod_t2)
        t = t2
    return a


def teichmueller_lift_newton(a0_mod_p, p, k_max):
    """
    用 Hensel-Newton 直接解 f(x)=x^{p-1}-1，起点 x0≡a0_mod_p (mod p)，得到 Teichmüller 代表。
    """
    x = int(a0_mod_p % p)
    if x % p == 0:
        raise ValueError("teichmueller_lift_newton expects a unit mod p")
    for k in range(2, k_max + 1):
        p_k = pow(p, k)
        p_km1 = pow(p, k - 1)
        f_mod = (powmod(x, p - 1, p_k) - 1) % p_k
        # f'(x) = (p-1) * x^{p-2} mod p
        fprime_mod_p = ((p - 1) * powmod(x, (p - 2) % (p - 1) if p > 2 else 0, p)) % p
        inv_fprime = pow(fprime_mod_p, -1, p)
        t = (-(f_mod // p_km1) * inv_fprime) % p
        x = (x + t * p_km1) % p_k
    return x


def construct_solution_via_padic_logexp(c, e, p, k_max):
    """
    使用 p 进 log/exp + 正确的 Teichmüller 分解构造一组解：
    - ω(c) = teich(c mod p)
    - ⟨c⟩ = c * ω(c)^{-1} ∈ 1+pZ_p
    - ⟨x⟩ = exp(e^{-1} * log⟨c⟩)
    - ω(x) 需满足 ω(x)^e = ω(c)（在 μ_{p-1} 中），若 m=gcd(e,p-1)>1，则有 m 个分支；这里选 j=0 分支：
        设 e=e1*m, r=(p-1)/m, 取 d ≡ e1^{-1} (mod r)，则 ω(x) = ω(c)^d
    - 返回 x = ω(x) * ⟨x⟩ (mod p^k)
    注意：返回的分支可能与 Hensel 流程的 x 不同，但应满足 x^e ≡ c (mod p^k)。
    """
    modulus = pow(p, k_max)
    # ω(c) in F_p and Teichmüller lift
    Fp = GF(p)
    wc_fp = Fp(int(c % p))
    if int(wc_fp) == 0:
        raise ValueError("c must be a unit modulo p for p-adic log/exp method")
    # 使用 Newton 版本更稳健
    w_c = teichmueller_lift_newton(int(wc_fp), p, k_max)
    # principal unit part
    main = (c * invmod(w_c, modulus)) % modulus
    R = Zp(p, prec=k_max + 2)
    main_p = R(main)
    e_inv = R(e).inverse()
    x_main = (main_p.log() * e_inv).exp()
    try:
        x_main_int = int(x_main)
    except Exception:
        x_main_int = int(x_main.lift())
    # choose ω(x) via field e-th root branch: ω(x)^e = ω(c)
    wx_fp = wc_fp.nth_root(e)
    w_x = teichmueller_lift_newton(int(wx_fp), p, k_max)
    return (w_x * (x_main_int % modulus)) % modulus

# --- Main Execution ---

def run_amm(n_val=None, e_val=None, c_val=None, q_val=None, verify=True, pattern=b'flag', use_newton_threshold=8):
    """
    运行 AMM 求解器，寻找 x 满足 x^e ≡ c (mod n)，其中 n = q^k。
    - n_val, e_val, c_val, q_val: 可选参数，若
        提供则覆盖全局变量 n, e, c, q。
    - verify: 是否进行结果验证和日志输出。
    - pattern: 查找 flag 的字节模式。
    - use_newton_threshold: k 超过该值时使用 Newton 提升，否则使用
        线性提升。
    返回包含 k, m, x, zeta, modulus, hit, steps 的字典。
    """
    n0 = n if n_val is None else n_val
    e0 = e if e_val is None else e_val
    c0 = c if c_val is None else c_val
    q0 = q if q_val is None else q_val
    k0 = v_p(n0, q0)

    t0 = time.time()
    x0, z0, m0 = find_field_root_and_unity(c0 % q0, q0, e0)
    t1 = time.time()
    logger.info(f'[timing] find_one_root+unity: {(t1 - t0):.3f}s, m={m0}, k={k0}')

    t2 = time.time()
    if k0 > use_newton_threshold:
        logger.info('Hensel lifting (Newton doubling)...')
        xk = hensel_lift_root_newton(c0, e0, q0, k0, x0)
        zk = hensel_lift_unity_newton(m0, q0, k0, z0)
    else:
        logger.info('Hensel lifting (linear-by-level)...')
        xk = hensel_lift_root_linear(c0, e0, q0, k0, x0)
        zk = hensel_lift_unity_linear(m0, q0, k0, z0)
    t3 = time.time()
    logger.info(f'[timing] hensel_lift total: {(t3 - t2):.3f}s')

    modk = pow(q0, k0)

    if verify:
        import random
        ok = 0
        for _ in range(5):
            j = random.randrange(0, max(1, min(m0, 1000)))
            val = (powmod(zk, j, modk) * (xk % modk)) % modk
            if powmod(val, e0, modk) == (c0 % modk):
                ok += 1
        logger.info(f'[verify] sampled solutions mod q^k: {ok}/5 OK')
        try:
            xl = hensel_lift_root_linear(c0, e0, q0, k0, x0)
            zl = hensel_lift_unity_linear(m0, q0, k0, z0)
            xn = hensel_lift_root_newton(c0, e0, q0, k0, x0)
            zn = hensel_lift_unity_newton(m0, q0, k0, z0)
            logger.info(f'[verify] lift(single)= {xl==xn}, lift(unity)= {zl==zn}')
        except Exception as _e:
            logger.warning(f'[verify] lift compare skipped: {_e}')
        logger.info('[verify] zeta^m check: ' + ('OK' if powmod(zk, m0, modk) == 1 % modk else 'FAIL'))
        logger.info('[verify] k=v_q(n): ' + ('OK' if (n0 % pow(q0, k0) == 0 and n0 % pow(q0, k0+1) != 0) else 'FAIL'))
        try:
            z_teich = teichmueller_lift_doubling(z0, q0, k0)
            logger.info('[verify] zeta teich vs newton equal: ' + str(z_teich == zk))
        except Exception as _e:
            logger.warning(f'[verify] zeta teich compare skipped: {_e}')
        try:
            x_padic = construct_solution_via_padic_logexp(c0, e0, q0, k0)
            eq_e = (powmod(x_padic % modk, e0, modk) == (c0 % modk))
            logger.info('[verify] x via p-adic log/exp satisfies equation: ' + str(eq_e))
            logger.info('[verify] x via p-adic log/exp equals Hensel x: ' + str((x_padic % modk) == (xk % modk)))
        except Exception as _e:
            logger.warning(f'[verify] p-adic log/exp compare skipped: {_e}')

    logger.info('Checking solutions for the flag...')
    byte_len = (modk.bit_length() + 7) // 8
    primary = pattern + b'{' if pattern == b'flag' else pattern
    fallback = pattern

    if _GMP:
        mod_mpz = mpz(modk)
        res_mpz = mpz(xk)
        z_mpz = mpz(zk)
    else:
        mod_mpz = None
        res_mpz = xk
        z_mpz = zk

    t4 = time.time()
    steps = 0
    hit = None
    for _ in range(m0):
        steps += 1
        try:
            iv = int(res_mpz)
            b = iv.to_bytes(byte_len, 'big')
            if (primary in b) or (fallback in b):
                logger.info(f'[+] FLAG FOUND: {b}')
                logger.info(f'[timing] steps_to_hit: {steps}')
                hit = b
                break
        except (ValueError, OverflowError):
            pass
        if _GMP:
            res_mpz = (res_mpz * z_mpz) % mod_mpz
        else:
            res_mpz = (res_mpz * z_mpz) % modk
    t5 = time.time()
    logger.info(f'[timing] enumerate+check: {(t5 - t4):.3f}s')

    return {
        'k': k0,
        'm': m0,
        'x': xk,
        'zeta': zk,
        'modulus': modk,
        'hit': hit,
        'steps': steps
    }

# 默认执行
_ = run_amm()

print('Finished.')
