from ctypes import sizeof
import random 
import numpy as np
import pandas as pd
import time
import os

import re

from MSS_system import *

# =======

# # AES_sbox2poly：using modulo 257 as an example, we get the polynomial:

AES_modulo = 257

AES_sbox2poly = """
99x^255 + 148x^254 + 19x^253 + 62x^252 + 210x^251 + 
78x^250 + 235x^249 + 192x^248 + 223x^247 + 131x^246 + 53x^245 + 5x^244 + 43x^243 + 40x^242 + 4x^241 + 
250x^240 + 170x^239 + 59x^238 + 130x^237 + 99x^236 + 252x^235 + 231x^234 + 142x^233 + 7x^232 + 118x^231 + 
80x^230 + 133x^229 + 156x^228 + 34x^227 + 161x^226 + 185x^225 + 100x^224 + 75x^223 + 174x^222 + 144x^221 + 
4x^220 + 111x^219 + 249x^218 + 250x^217 + 41x^216 + 218x^215 + 156x^214 + 197x^213 + 235x^212 + 250x^211 + 
107x^210 + 41x^209 + 237x^208 + 54x^207 + 32x^206 + 219x^205 + 113x^204 + 214x^203 + 217x^202 + 28x^201 + 
239x^200 + 230x^199 + 119x^198 + 3x^197 + 123x^196 + 179x^195 + 162x^194 + 50x^193 + 35x^192 + 97x^191 + 
53x^190 + 235x^189 + 249x^188 + 208x^187 + 167x^186 + 54x^185 + 207x^184 + 203x^183 + 22x^182 + 238x^181 + 
127x^180 + 33x^179 + 184x^178 + 168x^177 + 143x^176 + 192x^175 + 4x^174 + 10x^173 + 16x^172 + 203x^171 + 
168x^170 + 35x^169 + 238x^168 + 165x^167 + 211x^166 + 179x^165 + 160x^164 + 220x^163 + 55x^162 + 124x^161 + 
229x^160 + 211x^159 + 158x^158 + 97x^157 + 190x^156 + 230x^155 + 171x^154 + 215x^153 + 68x^152 + 104x^151 + 
229x^150 + 97x^149 + 125x^148 + 205x^147 + 13x^146 + 242x^145 + 67x^144 + 239x^143 + 26x^142 + 118x^141 + 
16x^140 + 94x^139 + 202x^138 + 23x^137 + 209x^136 + 123x^135 + 90x^134 + 11x^133 + 72x^132 + 84x^131 + 
93x^130 + 173x^129 + 11x^128 + 246x^127 + 52x^126 + 51x^125 + 54x^124 + 152x^123 + 204x^122 + 192x^121 + 
219x^120 + 42x^119 + 95x^118 + 172x^117 + 106x^116 + 132x^115 + 29x^114 + 213x^113 + 190x^112 + 227x^111 + 
171x^110 + 96x^109 + 73x^108 + 30x^107 + 75x^106 + 190x^105 + 193x^104 + 174x^103 + 147x^102 + 247x^101 + 
235x^100 + 120x^99 + 69x^98 + 246x^97 + 174x^96 + 72x^95 + 91x^94 + 34x^93 + 234x^92 + 251x^91 + 
44x^90 + 71x^89 + 53x^88 + 86x^87 + 17x^86 + 98x^85 + 97x^84 + 70x^83 + 235x^82 + 46x^81 + 
171x^80 + 43x^79 + 94x^78 + 152x^77 + 203x^76 + 153x^75 + 4x^74 + 219x^73 + 155x^72 + 209x^71 + 
123x^70 + 162x^69 + 117x^68 + 251x^67 + 185x^66 + 12x^65 + 154x^64 + 132x^63 + 143x^62 + 109x^61 + 
209x^60 + 194x^59 + 24x^58 + 66x^57 + 89x^56 + 111x^55 + 205x^54 + 183x^53 + 162x^52 + 85x^51 + 
109x^50 + 157x^49 + 103x^48 + 195x^47 + 116x^46 + 242x^45 + 252x^44 + 74x^43 + 2x^42 + 98x^41 + 
105x^40 + 247x^39 + 120x^38 + 102x^37 + 21x^36 + 125x^35 + 7x^34 + 44x^33 + 172x^32 + 106x^31 + 
213x^30 + 233x^29 + 5x^28 + 63x^27 + 62x^26 + 55x^25 + 90x^24 + 223x^23 + 85x^22 + 7x^21 + 
227x^20 + 245x^19 + 218x^18 + 37x^17 + 197x^16 + 99x^15 + 114x^14 + 91x^13 + 186x^12 + 252x^11 + 
163x^10 + 116x^9 + 101x^8 + 107x^7 + 98x^6 + 83x^5 + 154x^4 + 152x^3 + 209x^2 + 231x^1 + 
86x^0
"""

AES_sbox2poly_test = """
1x^10 + 1x^9 + 1x^8 + 1x^7 + 1x^6 + 1x^5 + 1x^4 + 1x^3 + 1x^2 + 1x^1 + 
1x^0
"""

# =======

# Global Parameters

# Mersenne Prime 4th(7) 5th(13) 6th(17) 7th(19) = 127, 8191, 131071, 524287
PRIME = 2**19 - 1

# L的最小值，必需比資料最大值(包括計算後)來的大。
L_Min = 3000

random_size = 100

# ====

u, T = 2,2

# # Basic numbers：協助運算的已知 secret。
# scalar_multiplication 預設 B_K[0] = 1 協助 data upload。
B_K = [ 1 , 0 ]
B_t = [ 1 , 1 ]
b_k = len(B_K)

basic_number_one_index = 0
basic_number_zero_index = 1

# =======

def MSS_system_init(poly_coeff):

    # # K：multi-secret list ( K < PRIME )，PRIME = 524287。
    K = poly_coeff

    # # t：threshold list for each secret (1 <= t <= n)，threshold 可非固定。
    # 為求計算方便，使用固定的 threshold，size 與 secret 一樣多。
    t = [T] * len(K)

    # 加入 Basic numbers：協助運算的已知 secret。
    K = B_K + K
    t = B_t + t

    # 建立 Dealer
    dealer = Dealer(u, K, t, None)

    # User 收到 獨立id = x座標。
    clients = []        
    for i in range(u):
        clients.append(Client(i))

    # 開始分發 User share & 製作 雙雲Server 的 public share。
    MSS = dealer.distribute(clients)

    del dealer

    return MSS, clients

def MSS_AES_sbox(MSS, participant, coeff_size, x):

    MSS.clear()

    # 上傳 x，製作 x 的 MSS_share。
    x_index = MSS.scalar_multiplication(participant, basic_number_one_index , x)

    # print("- x:", x, "，x_index:", x_index, "，reconstruct_x:", MSS.reconstruct_MSS_Secret(pool, x_index))

    poly_id = basic_number_zero_index
    for i in range(coeff_size):
        
        if(i % 100 == 0):
            print("-> coeff_index:", coeff_size - i)

        c = b_k + i
        
        poly_id = MSS.addition(participant, poly_id , c)

        if(i == coeff_size - 1):
            break;
        
        poly_id = MSS.multiplication(participant, poly_id , x_index)

        # print("- index:", coeff_size - i, "，poly:", MSS.reconstruct_MSS_Secret(pool, poly_id), "，c:", MSS.reconstruct_MSS_Secret(pool, c))

    result = MSS.reconstruct_MSS_Secret(pool, poly_id)

    # 計算結束後，將 record 清理乾淨，節省儲存空間。
    MSS.clear()

    return result % AES_modulo

# =======

def get_poly_coeff(polynomial_string, max_index):

    # 使用正則表達式找到每個係數
    terms = re.findall(r'(\d+)x\^(\d+)', polynomial_string)
    
    # 將係數填入list中，並補 0 以保證 list長度相同
    coefficients = [0] * max_index
    for coefficient, exponent in terms:
        coefficients[int(exponent)] = int(coefficient)

    coefficients.reverse()
    
    return coefficients

if __name__ == '__main__':

    epoch = 10

    print('', file=open('application_2__log.txt', 'w'))

    print('\n===========\n', file=open('application_2__log.txt', 'a+'))

    # poly_text = AES_sbox2poly_test
    # poly_size = 11

    poly_text = AES_sbox2poly
    poly_size = 256

    coeff = get_poly_coeff(poly_text, poly_size)

    # print("\ncoeff = ", coeff)
    # print("- sum = ", sum(coeff))
    # print("- sum under AES_modulo = ", sum(coeff) % AES_modulo)

    print('poly_text = \n', poly_text, file=open('application_2__log.txt', 'a+'))
    print('- poly_size = ', poly_size, file=open('application_2__log.txt', 'a+'))
    print('\n- AES_modulo = ', AES_modulo, file=open('application_2__log.txt', 'a+'))

    coeff_size = len(coeff)

    MSS, clients = MSS_system_init(coeff)     # 建立 MSS 系統

    pool = random.sample(clients, T)

    print("\n===========\n")
    print('\n===========\n', file=open('application_2__log.txt', 'a+'))

    print('Function time cost：')
    print('Function time cost：', file=open('application_2__log.txt', 'a+'))

    time1 = time.time()

    for i in range(epoch):
        operation_index = MSS.multiplication(pool, 1 , 2)

    time2 = time.time()

    time_cost = time2 - time1

    print("- time_cost [ MSS.multiplication() ]:", time_cost / epoch)
    print("- time_cost [ MSS.multiplication() ]:", time_cost / epoch, file=open('application_2__log.txt', 'a+'))

    time1 = time.time()

    for i in range(epoch):
        operation_index = MSS.addition(pool, 1 , 2)

    time2 = time.time()

    time_cost = time2 - time1

    print("- time_cost [ MSS.addition() ]:", time_cost / epoch)
    print("- time_cost [ MSS.addition() ]:", time_cost / epoch, file=open('application_2__log.txt', 'a+'))

    print("\n===========\n")
    print('\n===========\n', file=open('application_2__log.txt', 'a+'))
    
    input = 1
    
    output = MSS_AES_sbox(MSS, pool, coeff_size, input)

    print("\n", "- MSS_AES_sbox(", input , ") = " , output, "\n")
    print("- MSS_AES_sbox(", input , ") = " , output, file=open('application_2__log.txt', 'a+'))

    input = 0
    
    output = MSS_AES_sbox(MSS, pool, coeff_size, input)

    print("\n", "- MSS_AES_sbox(", input , ") = " , output, "\n")
    print("- MSS_AES_sbox(", input , ") = " , output, file=open('application_2__log.txt', 'a+'))

    time1 = time.time()

    for i in range(epoch):
        MSS_AES_sbox(MSS, pool, coeff_size, i)

    time2 = time.time()

    time_cost = time2 - time1

    print("\n=> time_cost [ MSS_AES_sbox() ]:", time_cost / epoch)
    print("\n=> time_cost [ MSS_AES_sbox() ]:", time_cost / epoch, file=open('application_2__log.txt', 'a+'))

    print("\n===========\n")
    print('\n===========\n', file=open('application_2__log.txt', 'a+'))

# =======