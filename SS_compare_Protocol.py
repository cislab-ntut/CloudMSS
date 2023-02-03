import numpy as np

import random

from secret_sharing import generateShares, reconstructSecret

# Mersenne Prime 4th 5th 7th
l_Min = 2**7 - 1
l_Max = 2**13 - 1
PRIME = 2**19 - 1

def share_list_addition(a, b):
    c = []
    for i in range(len(a)):
        x = a[i][0]
        for j in range(len(b)):
            if x == b[j][0]:
                y = (a[i][1] + b[j][1]) % PRIME
                break
        c.append([ x , y ])
    return c

def share_list_minus(a, b):
    c = []
    for i in range(len(a)):
        x = a[i][0]
        for j in range(len(b)):
            if x == b[j][0]:
                y = (a[i][1] - b[j][1]) % PRIME
                break
        c.append([ x , y ])
    return c

def share_list_constant_multiplication(a, value):
    c = []
    for i in range(len(a)):
        c.append([ a[i][0], (a[i][1] * value) % PRIME ])
    return c

# L的最小值，必需比資料最大值(包括計算後)來的大。
L_Min = 3000

n , t = (6, 3)

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
    de = share_list_constant_multiplication(share_e, d)
    bd = share_list_constant_multiplication(share_b, d)
    ae = share_list_constant_multiplication(share_a, e)
    de_bd = share_list_addition(de, bd)
    ae_c = share_list_addition(ae, share_c)

    xy = share_list_addition(de_bd, ae_c)
    
    return xy

def compare(a_share, b_share):

    r2 = random.randint(1, 100)
    r1 = random.randint(1, 50)

    # 2lr + r' < PRIME
    L_max = round(((PRIME - r2) / r1) / 2)
    l = random.randint(L_Min, L_max)

    l_share = generateShares(n, t, l)
    r1_share = generateShares(n, t, r1)
    r2_share = generateShares(n, t, r2)

    # =============

    a_MINUS_b = share_list_minus(a_share , b_share)
    
    a_MINUS_b_ADD_l = share_list_addition(a_MINUS_b , l_share)
    
    m = reconstructSecret(a_MINUS_b_ADD_l)
    l = reconstructSecret(l_share)

    # 使比較的基準對象，從還原的數值l，轉變成 h，不會產生隱私洩漏問題。
    # # 否則，對於知道 差值m 的 資料a或b之擁有者，能輕易推得另一個數字之值。
    m_MUL_r1 = multiply(a_MINUS_b_ADD_l , r1_share)
    m_MUL_r1_add_r2 = share_list_addition(m_MUL_r1 , r2_share)
    
    l_MUL_r1 = multiply(l_share , r1_share)
    l_MUL_r1_add_r2 = share_list_addition(l_MUL_r1 , r2_share)
    
    s = reconstructSecret(m_MUL_r1_add_r2)
    h = reconstructSecret(l_MUL_r1_add_r2)

    # print("s:" , s)
    # print("h:" , h)

    return s > h

if __name__=='__main__':

    print("\n====\n")

    # Test with comparison
    correct = True
    for i in range(10000):
        record = []

        secret1 = np.random.randint(0 , 1000)
        secret2 = np.random.randint(0 , 1000)
        
        share1 = generateShares(n, t, secret1)
        share2 = generateShares(n, t, secret2)

        result = compare(share1, share2)
        
        if((secret1 > secret2) != result):
            correct = False
            print("Error: secret1={} secret2={} result={}" .format(secret1 , secret2, result))

    if(correct):
        print("Comparison all correct！")
    else:
        print("Some Comparison wrong！")

    print("\n====\n")
