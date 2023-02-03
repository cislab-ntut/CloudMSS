import numpy as np
from secret_sharing import generateShares, reconstructSecret

# Mersenne Prime 7th
PRIME = 2**19 - 1

def share_list_addition(a, b):
    c = []
    for i in range(len(a)):
        c.append([ a[i][0], (a[i][1] + b[i][1]) % PRIME ])
    return c

def share_list_minus(a, b):
    c = []
    for i in range(len(a)):
        c.append([ a[i][0], (a[i][1] - b[i][1]) % PRIME ])
    return c

def share_list_constant_multiplication(a, value):
    c = []
    for i in range(len(a)):
        c.append([ a[i][0], (a[i][1] * value) % PRIME ])
    return c

def multiply(share_x , share_y):

    a = np.random.randint(100)
    b = np.random.randint(100)
    c = a * b

    share_a = generateShares(n, t, a)
    share_b = generateShares(n, t, b)
    share_c = generateShares(n, t, c)

    share_d = share_list_minus(share_x, share_a)
    share_e = share_list_minus(share_y, share_b)

    d = reconstructSecret(share_d) % PRIME
    e = reconstructSecret(share_e) % PRIME

    # xy = d * e + share_list_constant_multiplication(share_b, d) + share_list_constant_multiplication(share_a, e) + share_c
    
    xy = d * e + reconstructSecret( share_list_addition( share_list_addition( share_list_constant_multiplication(share_b, d) , share_list_constant_multiplication(share_a, e) ) , share_c ) )

    return xy % PRIME
    
if __name__=='__main__':

    print("\n====\n")

    n,t = 10,7
    
    secret1 = 4
    secret2 = 6
    print('Secret 1:' , secret1)
    print('Secret 2:' , secret2)

    ss1 = generateShares(n, t, secret1)
    ss2 = generateShares(n, t, secret2)
    print('Share 1:' , ss1 , ', reconstruct 1:' , reconstructSecret(ss1))
    print('Share 2:' , ss2 , ', reconstruct 1:' , reconstructSecret(ss2))
    print()

    SS_poly_Multiplication = multiply(ss1, ss2)
    print('SS_poly_Multiplication:' , SS_poly_Multiplication)

    print("\n====\n")
