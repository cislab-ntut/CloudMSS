from asyncio.windows_events import NULL

import random 
import numpy as np
import pandas as pd
import time

from secret_sharing import generateShares, reconstructSecret

from multi_secret_sharing import generate_Participant_Share, generate_Public_Shares

# ====

# Global Parameters

PRIME = 2**13 - 1   # Mersenne Prime 5th = 8191

random_size = 100

# Basic Tools

def _extended_gcd(a, b):
    """
    Division in integers modulus p means finding the inverse of the denominator modulo p. 
    < Note: inverse of A is B such that (A * B) % p == 1 >
    this can be computed via extended Euclidean algorithm (擴展歐幾里得算法，又叫「輾轉相除法」): 
    http://en.wikipedia.org/wiki/Modular_multiplicative_inverse#Computation
    """
    x = 0
    last_x = 1
    y = 1
    last_y = 0

    while b != 0:
        quot = a // b
        a, b = b, a % b

        x, last_x = last_x - quot * x, x
        y, last_y = last_y - quot * y, y
        
    return last_x, last_y

def _divmod(num, den, p):
    # Compute num / den modulo prime p
    invert, _ = _extended_gcd(den, p)
    return num * invert

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

# ====

class Dealer:
        
    def __init__(self, Participants, secrets, secrets_participant_threshold):
        self.n = Participants
        self.data = secrets
        self.t = secrets_participant_threshold

    def distribute(self, clients):

        MSS = MSS_system(self.n, len(self.data), self.t, clients)
        
        server_1 = MSS.call_server_1()
        server_2 = MSS.call_server_2()

        Participant_Share = generate_Participant_Share(self.n)

        for i in range(len(Participant_Share)):
            clients[i].get_share(Participant_Share[i])

        pseudo_secret_1 = np.random.randint(1, random_size, size = len(self.data))
        pseudo_secret_2 = pseudo_secret_1 * np.array(self.data) % PRIME

        Public_Shares_1 = generate_Public_Shares(Participant_Share, pseudo_secret_1, self.t)
        Public_Shares_2 = generate_Public_Shares(Participant_Share, pseudo_secret_2, self.t)
        
        server_1.get_share(Public_Shares_1)
        server_2.get_share(Public_Shares_2)

        print("Participants:" , self.n)
        print("Participant_Share:", Participant_Share)
        print()
        print("Secrets:" , self.data)
        print("pseudo_secret_1:" , pseudo_secret_1)
        print("pseudo_secret_2:" , pseudo_secret_2)

        return MSS

class Client:
    
    def __init__(self, id):
        self.id = id
        self.share = None

    def get_share(self, Participant_Share):
        self.share = Participant_Share

    def sent_id(self):
        return self.id

    def sent_share(self, share_a):
        x = self.share[0]
        y = (self.share[1] - share_a[1]) % PRIME
        return [ x , y ]
    
    def sent_new_share(self, share_a, share_b, share_c, d, e):
        x = self.share[0]
        y = ( e * (self.share[1] - share_a[1]) + d * share_b[1] + e * share_a[1] + share_c[1] ) % PRIME
        return [ x , y ]

class MSS_system:

    def __init__(self, n, k, t, clients):
        self.n = n                      # 參與者 數量
        self.k = k                      # secret 數量
        self.t = t                      # secret threshold

        self.server_1 = self.Server_1(self)
        self.server_2 = self.Server_2(self)
        self.RG = self.Randomness_Generator(self, clients)

        self.operation_record = {}      # operation 紀錄

    def call_server_1(self):
        return self.server_1
    
    def call_server_2(self):
        return self.server_2

    def call_RG(self):
        return self.RG

    def call_global_parameter(self):
        return self.n, self.k, self.t

    def print_operation_record(self):

        print("Operation Record:")
        print()

        print("- Original Secret index: 0 -", self.k - 1)
        print("- PRIME: ", PRIME)
        print()

        operation_record = self.operation_record

        exception_keywords = ["participants", "operand_a", "operand_b"]
        
        print("- Operation record：")

        if len(operation_record) == 0:
            print("None operation！")
        else:
            for operation_id in operation_record:
                print("-- index:" , operation_id , end = ' ')

                for key in operation_record[operation_id]:
                    if key not in exception_keywords:
                        print(" , " , key , ":" , operation_record[operation_id][key] , end = ' ')

                print(" , participants threshold:" , self.t[operation_id])

    class Server_1:
        
        def __init__(self, MSS_system):
            self.MSS = MSS_system
            self.share = None
            # self.operation_record = {}      # operation 紀錄
        
        def get_share(self, Public_Share):
            self.share = Public_Share
        
        def sent_share(self, i, num, share_a_list):
            new_share = share_list_minus(self.share[i], share_a_list)
            return new_share[0 : num]

        def sent_new_share(self, i, num, share_a_list, share_b_list, share_c_list, d, e):
            new_share_1 = share_list_constant_multiplication( share_list_minus(self.share[i], share_a_list), e )
            new_share_2 = share_list_constant_multiplication( share_b_list, d )
            new_share_3 = share_list_constant_multiplication( share_a_list, e )
            new_share_4 = share_list_addition( share_list_addition(new_share_1, new_share_2), share_list_addition(new_share_3, share_c_list) )
            return new_share_4[0 : num]
        
    class Server_2:
        
        def __init__(self, MSS_system):
            self.MSS = MSS_system
            self.share = None
            self.operation_record = {}      # operation 紀錄
        
        def get_share(self, Public_Share):
            self.share = Public_Share

        def sent_share(self, i, num, share_a_list):
            new_share = share_list_minus(self.share[i], share_a_list)
            return new_share[0 : num]
        
        def sent_new_share(self, i, num, share_a_list, share_b_list, share_c_list, d, e):
            new_share_1 = share_list_constant_multiplication( share_list_minus(self.share[i], share_a_list), e )
            new_share_2 = share_list_constant_multiplication( share_b_list, d )
            new_share_3 = share_list_constant_multiplication( share_a_list, e )
            new_share_4 = share_list_addition( share_list_addition(new_share_1, new_share_2), share_list_addition(new_share_3, share_c_list) )
            return new_share_4[0 : num]

    # ====

    class Randomness_Generator:

        def __init__(self, MSS_system, clients):
            self.MSS = MSS_system
            self.clients = clients
            self.randomness_record = {}     # randomness 紀錄

        def poly_randomness(self, i):

            n, k, t = self.MSS.call_global_parameter()

            # ====

            r = np.random.randint(1, random_size)
            # print('r:' , r)

            share_r1 = generateShares(2 * n + 1 - t[i] , t[i] , r)
            share_r2 = generateShares(2 * n + 1 - t[i] , t[i] , r)

            # ====

            if i < k:               # 非運算

                server_1 = self.MSS.call_server_1()

                a1 = np.random.randint(1, random_size)
                b1 = np.random.randint(1, random_size)
                c1 = (a1 * b1) % PRIME 

                share_a1 = generateShares(2 * n + 1 - t[i] , t[i] , a1)
                share_b1 = generateShares(2 * n + 1 - t[i] , t[i] , b1)
                share_c1 = generateShares(2 * n + 1 - t[i] , t[i] , c1)

                share_d1 = []
                for j in range(n):
                    share_d1.append(self.clients[j].sent_share(share_a1[j]))                # 模擬 client 收到各自的 random share 將自己的 share masked 再傳出。
                
                share_d1 = share_d1 + server_1.sent_share(i, 1, share_a1[ n : ])       # 模擬 client 收到各自的 random share 將自己的 share masked 再傳出。

                share_e1 = share_list_minus(share_r1, share_b1)

                d1 = reconstructSecret(share_d1) % PRIME
                e1 = reconstructSecret(share_e1) % PRIME

                randomness_index_1 = len(self.randomness_record)

                self.randomness_record[randomness_index_1] = {
                    # "r": r,
                    "share_a": share_a1,
                    "share_b": share_b1,
                    "share_c": share_c1,
                    # "share_d": share_d1,
                    "d": d1,
                    "e": e1,
                }

                # ====

                server_2 = self.MSS.call_server_2()

                a2 = np.random.randint(1, random_size)
                b2 = np.random.randint(1, random_size)
                c2 = (a2 * b2) % PRIME 

                share_a2 = generateShares(2 * n + 1 - t[i] , t[i] , a2)
                share_b2 = generateShares(2 * n + 1 - t[i] , t[i] , b2)
                share_c2 = generateShares(2 * n + 1 - t[i] , t[i] , c2)

                share_d2 = []
                for j in range(n):
                    share_d2.append(self.clients[j].sent_share(share_a2[j]))                # 模擬 client 收到各自的 random share 將自己的 share masked 再傳出。

                share_d2 = share_d2 + server_2.sent_share(i, 1, share_a2[ n : ])       # 模擬 client 收到各自的 random share 將自己的 share masked 再傳出。

                share_e2 = share_list_minus(share_r2, share_b2)

                d2 = reconstructSecret(share_d2) % PRIME
                e2 = reconstructSecret(share_e2) % PRIME

                randomness_index_2 = len(self.randomness_record)

                self.randomness_record[randomness_index_2] = {
                    # "r": r,
                    "share_a": share_a2,
                    "share_b": share_b2,
                    "share_c": share_c2,
                    # "share_d": share_d2,
                    "d": d2,
                    "e": e2,
                }
            
            else:                   # 運算
                operation_record = self.MSS.operation_record[i]
                
                participants = operation_record["participants"]
                
                ( a , a_randomness_index_1 , a_randomness_index_2 ) = operation_record["operand_a"]
                ( b,  b_randomness_index_1 , b_randomness_index_2 ) = operation_record["operand_b"]

                operation_shares_a_1, operation_shares_a_2 = self.MSS.collect_shares(participants , a , a_randomness_index_1, a_randomness_index_2)
                operation_shares_b_1, operation_shares_b_2 = self.MSS.collect_shares(participants , b , b_randomness_index_1, b_randomness_index_2)

                # operation_shares_a_1, operation_shares_a_2 = self.MSS.collect_shares(self.clients , a , a_randomness_index_1, a_randomness_index_2)
                # operation_shares_b_1, operation_shares_b_2 = self.MSS.collect_shares(self.clients , b , b_randomness_index_1, b_randomness_index_2)
                
                if operation_record["operation"] == "+":
                    pseudo_secret_a_1 = operation_record["pseudo_secret_a_1"]
                    pseudo_secret_b_1 = operation_record["pseudo_secret_b_1"]
                    
                    collect_shares_1 = share_list_constant_multiplication(operation_shares_a_1, pseudo_secret_b_1)
                    collect_shares_2 = share_list_addition(share_list_constant_multiplication(operation_shares_a_2, pseudo_secret_b_1) , share_list_constant_multiplication(operation_shares_b_2, pseudo_secret_a_1))
                elif operation_record["operation"] == "*":
                    pseudo_secret_a_1 = operation_record["pseudo_secret_a_1"]
                    pseudo_secret_b_2 = operation_record["pseudo_secret_b_2"]

                    collect_shares_1 = share_list_constant_multiplication(operation_shares_b_1, pseudo_secret_a_1)
                    collect_shares_2 = share_list_constant_multiplication(operation_shares_a_2, pseudo_secret_b_2)
                else:
                    raise Exception("Unrecognizable operation！")

                # ====

                a1 = np.random.randint(1, random_size)
                b1 = np.random.randint(1, random_size)
                c1 = (a1 * b1) % PRIME 

                share_a1 = generateShares(2 * n + 1 - t[i] , t[i] , a1)
                share_b1 = generateShares(2 * n + 1 - t[i] , t[i] , b1)
                share_c1 = generateShares(2 * n + 1 - t[i] , t[i] , c1)

                share_d1 = share_list_minus(collect_shares_1, share_a1)

                share_e1 = share_list_minus(share_r1, share_b1)

                d1 = reconstructSecret(share_d1) % PRIME
                e1 = reconstructSecret(share_e1) % PRIME

                randomness_index_1 = len(self.randomness_record)

                self.randomness_record[randomness_index_1] = {
                    # "r": r,
                    "share_a": share_a1,
                    "share_b": share_b1,
                    "share_c": share_c1,
                    # "share_d": share_d1,
                    "d": d1,
                    "e": e1,
                }

                # ====

                server_2 = self.MSS.call_server_2()

                a2 = np.random.randint(1, random_size)
                b2 = np.random.randint(1, random_size)
                c2 = (a2 * b2) % PRIME 

                share_a2 = generateShares(2 * n + 1 - t[i] , t[i] , a2)
                share_b2 = generateShares(2 * n + 1 - t[i] , t[i] , b2)
                share_c2 = generateShares(2 * n + 1 - t[i] , t[i] , c2)

                share_d2 = share_list_minus(collect_shares_2, share_a2)

                share_e2 = share_list_minus(share_r2, share_b2)

                d2 = reconstructSecret(share_d2) % PRIME
                e2 = reconstructSecret(share_e2) % PRIME

                randomness_index_2 = len(self.randomness_record)

                self.randomness_record[randomness_index_2] = {
                    # "r": r,
                    "share_a": share_a2,
                    "share_b": share_b2,
                    "share_c": share_c2,
                    # "share_d": share_d2,
                    "d": d2,
                    "e": e2,
                }
            
            return randomness_index_1, randomness_index_2

        def sent_randomness(self, index):
            randomness_record = self.randomness_record[index]

            share_a = randomness_record["share_a"]
            share_b = randomness_record["share_b"]
            share_c = randomness_record["share_c"]
            # share_d = randomness_record["share_d"]
            d = randomness_record["d"]
            e = randomness_record["e"]

            # return share_a, share_b, share_c, share_d, d, e
            return share_a, share_b, share_c, d, e

        def print_randomness_record(self):
            
            print("Randomness_Generator Record:")
            print()

            randomness_record = self.randomness_record

            exception_keywords = ["share_a", "share_b", "share_c", "share_de"]

            if len(randomness_record) == 0:
                print("None randomness！")
            else:
                for randomness_id in randomness_record:
                    print("-- index:" , randomness_id , end = ' ')

                    for key in randomness_record[randomness_id]:
                        if key not in exception_keywords:
                            print(" , " , key , ":" , randomness_record[randomness_id][key] , end = ' ')
                    
                    print()

    # ====

    # MSS Protocols

    def addition(self, participants, a, b):

        a_randomness_index_1, a_randomness_index_2 = self.RG.poly_randomness(a)

        collect_shares_a_1 , collect_shares_a_2 = self.collect_shares(participants , a , a_randomness_index_1, a_randomness_index_2)

        pseudo_secret_a_1 = reconstructSecret(collect_shares_a_1)

        b_randomness_index_1, b_randomness_index_2 = self.RG.poly_randomness(b)
        
        collect_shares_b_1 , collect_shares_b_2 = self.collect_shares(participants , b , b_randomness_index_1, b_randomness_index_2)

        pseudo_secret_b_1 = reconstructSecret(collect_shares_b_1)

        # ====
        
        operation_threshold = max(self.t[a] , self.t[b])
        
        operation_index = len(self.t)

        self.t = self.t + [ operation_threshold ]

        self.operation_record[operation_index] = {
            "info (index operation index)": str(a) + " + "+ str(b),
            "operation": "+",
            "operand_a": (a , a_randomness_index_1 , a_randomness_index_2),
            "operand_b": (b , b_randomness_index_1 , b_randomness_index_2),
            "pseudo_secret_a_1": pseudo_secret_a_1,
            "pseudo_secret_b_1": pseudo_secret_b_1,
            "participants": participants
        }

        return operation_index

    def multiplication(self, participants, a, b):

        a_randomness_index_1, a_randomness_index_2 = self.RG.poly_randomness(a)

        collect_shares_a_1 , collect_shares_a_2 = self.collect_shares(participants , a , a_randomness_index_1, a_randomness_index_2)

        pseudo_secret_a_1 = reconstructSecret(collect_shares_a_1)

        b_randomness_index_1, b_randomness_index_2 = self.RG.poly_randomness(b)
        
        collect_shares_b_1 , collect_shares_b_2 = self.collect_shares(participants , b , b_randomness_index_1, b_randomness_index_2)

        pseudo_secret_b_2 = reconstructSecret(collect_shares_b_2)

        # ==== 

        operation_threshold = max(self.t[a] , self.t[b])
        
        operation_index = len(self.t)

        self.t = self.t + [ operation_threshold ]

        self.operation_record[operation_index] = {
            "info (index operation index)": str(a) + " * "+ str(b),
            "operation": "*",
            "operand_a": (a , a_randomness_index_1 , a_randomness_index_2),
            "operand_b": (b , b_randomness_index_1 , b_randomness_index_2),
            "pseudo_secret_a_1": pseudo_secret_a_1,
            "pseudo_secret_b_2": pseudo_secret_b_2,
            "participants": participants
        }

        return operation_index

    def collect_shares(self, participants, i , randomness_index_1 , randomness_index_2):

        # share_a1, share_b1, share_c1, share_d1, d1, e1 = self.RG.sent_randomness(randomness_index_1)
        # share_a2, share_b2, share_c2, share_d2, d2, e2 = self.RG.sent_randomness(randomness_index_2)

        share_a1, share_b1, share_c1, d1, e1 = self.RG.sent_randomness(randomness_index_1)
        share_a2, share_b2, share_c2, d2, e2 = self.RG.sent_randomness(randomness_index_2)

        collect_shares_1 = []
        collect_shares_2 = []

        for client in  participants:
            id = client.sent_id()
            collect_shares_1.append( client.sent_new_share(share_a1[id], share_b1[id], share_c1[id], d1, e1) )     # 模擬 client 各自計算 share (= 乘上 randomness) 再傳出。
            collect_shares_2.append( client.sent_new_share(share_a2[id], share_b2[id], share_c2[id], d2, e2) )     # 模擬 client 各自計算 share (= 乘上 randomness) 再傳出。

        public_num = 0      # 需要由 server 給出的 public share 數量

        if len(participants) < self.t[i]:
            raise Exception("Need more participant share！")
        else: 
            public_num = self.n + 1 - len(participants)

        if i < self.k:      # 非運算
            collect_shares_1 = collect_shares_1 + self.server_1.sent_new_share( i, public_num, share_a1[ self.n : ], share_b1[ self.n : ], share_c1[ self.n : ], d1, e1 )
            collect_shares_2 = collect_shares_2 + self.server_2.sent_new_share( i, public_num, share_a2[ self.n : ], share_b2[ self.n : ], share_c2[ self.n : ], d2, e2 )
        else:               # 運算結果 
            operation_record = self.operation_record[i]

            for client in operation_record["participants"]:
                if client not in participants:
                    raise Exception("Need particular operation participants！")
            
            ( a , a_randomness_index_1 , a_randomness_index_2 ) = operation_record["operand_a"]
            ( b,  b_randomness_index_1 , b_randomness_index_2 ) = operation_record["operand_b"]

            operation_shares_a_1, operation_shares_a_2 = self.collect_shares(participants , a , a_randomness_index_1, a_randomness_index_2)
            operation_shares_b_1, operation_shares_b_2 = self.collect_shares(participants , b , b_randomness_index_1, b_randomness_index_2)
            
            if operation_record["operation"] == "+":
                pseudo_secret_a_1 = operation_record["pseudo_secret_a_1"]
                pseudo_secret_b_1 = operation_record["pseudo_secret_b_1"]
                
                collect_shares_1 = share_list_constant_multiplication(operation_shares_a_1, pseudo_secret_b_1)
                collect_shares_2 = share_list_addition(share_list_constant_multiplication(operation_shares_a_2, pseudo_secret_b_1) , share_list_constant_multiplication(operation_shares_b_2, pseudo_secret_a_1))
            elif operation_record["operation"] == "*":
                pseudo_secret_a_1 = operation_record["pseudo_secret_a_1"]
                pseudo_secret_b_2 = operation_record["pseudo_secret_b_2"]

                collect_shares_1 = share_list_constant_multiplication(operation_shares_b_1, pseudo_secret_a_1)
                collect_shares_2 = share_list_constant_multiplication(operation_shares_a_2, pseudo_secret_b_2)
            else:
                raise Exception("Unrecognizable operation！")
               
            new_collect_share_1_1 = share_list_constant_multiplication( share_list_minus(collect_shares_1, share_a1), e1 )
            # new_collect_share_1_1 = share_list_constant_multiplication( share_d1, e1 )
            new_collect_share_1_2 = share_list_constant_multiplication( share_b1, d1 )
            new_collect_share_1_3 = share_list_constant_multiplication( share_a1, e1 )
            new_collect_share_1_4 = share_list_addition( share_list_addition(new_collect_share_1_1, new_collect_share_1_2), share_list_addition(new_collect_share_1_3, share_c1) )

            new_collect_share_2_1 = share_list_constant_multiplication( share_list_minus(collect_shares_2, share_a2), e2 )
            # new_collect_share_2_1 = share_list_constant_multiplication( share_d2, e2 )
            new_collect_share_2_2 = share_list_constant_multiplication( share_b2, d2 )
            new_collect_share_2_3 = share_list_constant_multiplication( share_a2, e2 )
            new_collect_share_2_4 = share_list_addition( share_list_addition(new_collect_share_2_1, new_collect_share_2_2), share_list_addition(new_collect_share_2_3, share_c2) )

            collect_shares_1 = new_collect_share_1_4
            collect_shares_2 = new_collect_share_2_4

        return collect_shares_1 , collect_shares_2

    def reconstruct_Secret(self, participants, i):

        randomness_index_1, randomness_index_2 = self.RG.poly_randomness(i)

        collect_shares_1 , collect_shares_2 = self.collect_shares(participants, i , randomness_index_1, randomness_index_2)

        pseudo_secret_1 = reconstructSecret(collect_shares_1)
        pseudo_secret_2 = reconstructSecret(collect_shares_2)

        # print("i:", i , ", pseudo_secret_1:", pseudo_secret_1 , ", pseudo_secret_2:", pseudo_secret_2)

        secret = _divmod(pseudo_secret_2, pseudo_secret_1, PRIME)

        return secret % PRIME
    
# ====

if __name__ == '__main__':

    print("\n====\n")

    # # n：參與者數量
    # n = 1
    # n = 30
    n = 10

    # # K：multi-secret list ( K < PRIME )
    # K = [0, 1]
    K = [PRIME-1, 7000, 130, 20, 1, 0 , 1]

    # # k：secret 數量
    k = len(K)

    # # t：threshold list for each secret (t <= n)。
    # t = [0, 0]
    # t = [0, 11, 30, 4, 7, 13, 10]             # n = 30，threshold 非固定。
    t = [4, 4, 4, 4, 4, 4, 4]                   # n = 10，threshold 固定 (以求便於計算，可設置成非固定)。

    dealer = Dealer(n, K, t)

    clients = []        
    for i in range(n):
        clients.append(Client(i))

    MSS = dealer.distribute(clients)

    print("\n====\n")

    # 隨機參與者：人數 = t[i]。
    pool = random.sample(clients, t[1])


    # 測試：每筆secret還原狀況
    test_1 = True

    for i in range(k):
        reconstruct = MSS.reconstruct_Secret(pool, i)

        if(reconstruct != K[i]):
            test_1 = False
            print("Error: Secret = " , K[i] , ", Reconstruct = " , reconstruct)
    
    if (test_1 == True):
        print("Test_1: All reconstruct success.")
    else:
        print("Test_1: There is some reconstruct error.")

    print("\n====\n")

    # 測試：原始 secret 運算狀況

    i , j = 2 , 3

    operation_index_1 = MSS.addition(pool, i , j)
    reconstruct = MSS.reconstruct_Secret(pool, operation_index_1)

    print("Secret addition:" , MSS.reconstruct_Secret(pool, i) , "+" , MSS.reconstruct_Secret(pool, j))
    print("Reconstructed secret:", reconstruct)
    print()

    i , j = 2 , 3

    operation_index_2 = MSS.multiplication(pool, i , j)
    reconstruct = MSS.reconstruct_Secret(pool, operation_index_2)

    print("Secret multiplication:" , MSS.reconstruct_Secret(pool, i) , "*" , MSS.reconstruct_Secret(pool, j))
    print("Reconstructed secret:", reconstruct)
    print()

    print("\n====\n")

    # 測試：運算結果 與 原始 secret 運算狀況

    i , j = operation_index_1 , 3

    operation_index = MSS.addition(pool, i , j)
    reconstruct = MSS.reconstruct_Secret(pool, operation_index)

    print("Secret addition:" , MSS.reconstruct_Secret(pool, i) , "+" , MSS.reconstruct_Secret(pool, j))
    print("Reconstructed secret:", reconstruct)
    print()

    i , j = operation_index_1 , 3

    operation_index = MSS.multiplication(pool, i , j)
    reconstruct = MSS.reconstruct_Secret(pool, operation_index)

    print("Secret multiplication:" , MSS.reconstruct_Secret(pool, i) , "*" , MSS.reconstruct_Secret(pool, j))
    print("Reconstructed secret:", reconstruct)
    print()

    i , j = operation_index_1 , operation_index_2

    operation_index = MSS.addition(pool, i , j)
    reconstruct = MSS.reconstruct_Secret(pool, operation_index)

    print("Secret addition:" , MSS.reconstruct_Secret(pool, i) , "+" , MSS.reconstruct_Secret(pool, j))
    print("Reconstructed secret:", reconstruct)
    print()

    i , j = operation_index_1 , operation_index_2

    operation_index = MSS.multiplication(pool, i , j)
    reconstruct = MSS.reconstruct_Secret(pool, operation_index)
    
    print("Secret multiplication:" , MSS.reconstruct_Secret(pool, i) , "*" , MSS.reconstruct_Secret(pool, j))
    print("Reconstructed secret:", reconstruct)
    print()

    i , j = operation_index_1 , operation_index_2

    operation_index = MSS.addition(pool, i , j)
    reconstruct = MSS.reconstruct_Secret(pool, operation_index)

    print("Secret addition:" , MSS.reconstruct_Secret(pool, i) , "+" , MSS.reconstruct_Secret(pool, j))
    print("Reconstructed secret:", reconstruct)
    print()

    i , j = operation_index_1 , operation_index_2

    operation_index = MSS.multiplication(pool, i , j)
    reconstruct = MSS.reconstruct_Secret(pool, operation_index)
    
    print("Secret multiplication:" , MSS.reconstruct_Secret(pool, i) , "*" , MSS.reconstruct_Secret(pool, j))
    print("Reconstructed secret:", reconstruct)
    print()

    i , j = 5 , 6

    operation_index = MSS.addition(pool, i , j)
    reconstruct = MSS.reconstruct_Secret(pool, operation_index)
    
    print("Secret addition:" , MSS.reconstruct_Secret(pool, i) , "+" , MSS.reconstruct_Secret(pool, j))
    print("Reconstructed secret:", reconstruct)
    print()

    i , j = 5 , 6

    operation_index = MSS.multiplication(pool, i , j)
    reconstruct = MSS.reconstruct_Secret(pool, operation_index)
    
    print("Secret multiplication:" , MSS.reconstruct_Secret(pool, i) , "*" , MSS.reconstruct_Secret(pool, j))
    print("Reconstructed secret:", reconstruct)
    print()

    i , j = 6 , 5

    operation_index = MSS.addition(pool, i , j)
    reconstruct = MSS.reconstruct_Secret(pool, operation_index)
    
    print("Secret addition:" , MSS.reconstruct_Secret(pool, i) , "+" , MSS.reconstruct_Secret(pool, j))
    print("Reconstructed secret:", reconstruct)
    print()

    i , j = 6 , 5

    operation_index = MSS.multiplication(pool, i , j)
    reconstruct = MSS.reconstruct_Secret(pool, operation_index)
    
    print("Secret multiplication:" , MSS.reconstruct_Secret(pool, i) , "*" , MSS.reconstruct_Secret(pool, j))
    print("Reconstructed secret:", reconstruct)
    print()

    print("\n====\n")

    MSS.print_operation_record()

    # RG = MSS.call_RG()
    # RG.print_randomness_record()

    print("\n====\n")
