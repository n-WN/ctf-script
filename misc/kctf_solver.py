
#!/usr/bin/env python3
import base64, os, secrets, socket, sys, hashlib
try:
    import gmpy2
    HAVE_GMP = True
except ImportError:
    HAVE_GMP = False
VERSION = 's'
MODULUS = 2**1279-1
def python_sloth_root(x, diff, p):
    exponent = (p + 1) // 4
    for i in range(diff):
        x = pow(x, exponent, p) ^ 1
    return x
def gmpy_sloth_root(x, diff, p):
    exponent = (p + 1) // 4
    for i in range(diff):
        x = gmpy2.powmod(x, exponent, p).bit_flip(0)
    return int(x)
def sloth_root(x, diff, p):
    if HAVE_GMP: return gmpy_sloth_root(x, diff, p)
    else: return python_sloth_root(x, diff, p)
def encode_number(num):
    size = (num.bit_length() // 24) * 3 + 3
    return str(base64.b64encode(num.to_bytes(size, 'big')), 'utf-8')
def decode_number(enc):
    return int.from_bytes(base64.b64decode(bytes(enc, 'utf-8')), 'big')
def decode_challenge(enc):
    dec = enc.split('.')
    if dec[0] != VERSION: raise Exception('Unknown challenge version')
    return list(map(decode_number, dec[1:]))
def encode_challenge(arr):
    return '.'.join([VERSION] + list(map(encode_number, arr)))
def solve_challenge(chal):
    [diff, x] = decode_challenge(chal)
    y = sloth_root(x, diff, MODULUS)
    return encode_challenge([y])
def main():
    if len(sys.argv) != 3 or sys.argv[1] != 'solve': sys.exit(1)
    challenge = sys.argv[2]
    solution = solve_challenge(challenge)
    sys.stdout.write(solution)
if __name__ == "__main__":
    main()
