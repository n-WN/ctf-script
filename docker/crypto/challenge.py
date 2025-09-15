flag = b"flag{test}"

from Crypto.Util.number import *
from gmpy2 import *

def next_prime(x):
    while not isPrime(x):
        x += 1
    return x

success = 0

for i in range(20):
    coins = 27
    space = 0
    p = getPrime(64)
    delta = getRandomNBitInteger(30)
    q = next_prime(p + delta)
    N = p*q
    print("Welcome to my supermarket\n")
    while coins > 0:
        choice = input('give me your choice\n')
        if choice == '1':
            space = int(input("What size of house would you like to purchase?\n"))
            assert 1 <= space <= 10
            ls = [0] * space
            coins -= space * 5
            print(f'{coins} coins left\n')
        elif choice == '2':
            op = input()
            assert op in ['+', '-', '*', '//', '%', 'root']
            a, b, c= input().split('.')
            try:
                if op == 'root':
                    exec(f'{a}=iroot({b},{c})[0]')
                else:
                    exec(f'{a}={b}{op}{c}')
            except:exit
            if op in '+-':
                coins -= 1
            elif op in '*//%':
                coins -= 3
            else:
                coins -= 5
            print(f'{coins} coins left\n')

        elif choice == '3':
            state = 0
            print("One coin to check\n")
            coins -= 1
            print("You must have decorated a beautiful house.\n")
            assert coins >= 0
            for i in ls:
                if i > 1 and i < N and N%i == 0:
                    success += 1
                    state = 1
                    print(f'wonderful!, still {coins} coins left\n')
                    break
            if state:
                break

if success == 20:
    print(f'Congratulations! Here is your flag:{flag}\n')