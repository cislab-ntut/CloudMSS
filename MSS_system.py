import random 
import numpy as np
import pandas as pd
import time

from sklearn.metrics import jaccard_score

from multi_secret_sharing import generate_Participant_Share, generate_Public_Shares, reconstructSecret, _divmod

# ====

# Mersenne Prime 5th = 8191
PRIME = 2**13 - 1

random_size = 100

def share_list_addition(a, b):
    c = []
    for i in range(len(a)):
        c.append([ a[i][0], (a[i][1] + b[i][1]) % PRIME ])
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

    def distribute(self):
        Participant_Share = generate_Participant_Share(self.n)

        clients = []        
        for i in range(len(Participant_Share)):
            clients.append(Client(Participant_Share[i]))

        pseudo_secret_1 = np.random.randint(1, random_size, size = len(self.data))
        pseudo_secret_2 = pseudo_secret_1 * np.array(self.data) % PRIME

        Public_Shares_1 = generate_Public_Shares(Participant_Share, pseudo_secret_1, self.t)
        Public_Shares_2 = generate_Public_Shares(Participant_Share, pseudo_secret_2, self.t)
        
        server = Server(len(self.data), Public_Shares_1, Public_Shares_2, self.n, self.t)

        print("Participants:" , self.n)
        print("Participant_Share:")
        print(Participant_Share)
        print()
        print("Secrets:" , self.data)
        print("pseudo_secret_1:" , pseudo_secret_1)
        print("pseudo_secret_2:" , pseudo_secret_2)

        return server, clients

# ====

class Server:

    def __init__(self, k, Public_Shares_1, Public_Shares_2, Participants, secrets_participant_threshold):
        self.k = k                                  # secret 數量
        self.share_1 = Public_Shares_1
        self.share_2 = Public_Shares_2
        self.n = Participants                       # 參與者數量
        self.t = secrets_participant_threshold
        self.operation_record = {}                  # operation 紀錄

    def print_server_record(self):

        print("Server Record:")
        print()

        print("- Original Secret index: 0 -", self.k - 1)
        print("- PRIME: ", PRIME)
        print()

        operation_record = self.operation_record
        
        print("- Operation record：")

        if len(operation_record) == 0:
            print("None operation！")
        else:
            for operation_id in operation_record:
                print("-- index:" , operation_id , end = ' ')

                for key in operation_record[operation_id]:
                    if (key != "participants") and (key != "operand_a") and (key != "operand_b"):
                        print(" , " , key , ":" , operation_record[operation_id][key] , end = ' ')

                print(" , participants threshold:" , self.t[operation_id])

    # ====

    def collect_shares(self , participants , i):

        collect_shares = []

        for client in  participants:
            collect_shares.append(client.sent_share())

        public_num = 0

        if len(collect_shares) < self.t[i]:
            raise Exception("Need more participant share！")
        else: 
            public_num = self.n + 1 - len(collect_shares)

        if i < self.k:      # 還原 secret 內容
            collect_shares_1 = collect_shares + self.share_1[i][0 : public_num]
            collect_shares_2 = collect_shares + self.share_2[i][0 : public_num]
        else:               # 還原 運算結果
            operation_record = self.operation_record[i]

            for client in operation_record["participants"]:
                if client not in participants:
                    raise Exception("Need particular operation participants！")
            
            a = operation_record["operand_a"]
            b = operation_record["operand_b"]

            operation_shares_a_1, operation_shares_a_2 = self.collect_shares(participants , a)
            operation_shares_b_1, operation_shares_b_2 = self.collect_shares(participants , b)
            
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

        return collect_shares_1 , collect_shares_2

    # ====

    def _reconstruct_Secret(self, participants, i):

        collect_shares_1 , collect_shares_2 = self.collect_shares(participants, i)

        pseudo_secret_1 = reconstructSecret(collect_shares_1)
        pseudo_secret_2 = reconstructSecret(collect_shares_2)

        secret = _divmod(pseudo_secret_2, pseudo_secret_1, PRIME)

        return secret % PRIME

    # mode：{ 1 , 2 } => select either pseudo_secret_1 or pseudo_secret_2 would be return.
    def reconstruct_pseudo_secret(self, participants, i, mode):
        
        collect_shares_1 , collect_shares_2 = self.collect_shares(participants, i)

        pseudo_secret_1 = reconstructSecret(collect_shares_1)
        pseudo_secret_2 = reconstructSecret(collect_shares_2)

        if mode == 1:
            return pseudo_secret_1 % PRIME
        elif mode == 2:
            return pseudo_secret_2 % PRIME
        else:
            raise Exception("Unrecognizable mode of pseudo_secret reconstruction！")
    
    # ====

    def addition(self, participants, a, b):

        pseudo_secret_a_1 = self.reconstruct_pseudo_secret(participants, a, 1)
        pseudo_secret_b_1 = self.reconstruct_pseudo_secret(participants, b, 1)
        
        operation_threshold = max(self.t[a] , self.t[b])
        
        operation_index = len(self.t)

        self.t = self.t + [ operation_threshold ]

        self.operation_record[operation_index] = {
            "info (index operation index)": str(a) + " + "+ str(b),
            "operation": "+",
            "operand_a": a,
            "operand_b": b,
            "pseudo_secret_a_1": pseudo_secret_a_1,
            "pseudo_secret_b_1": pseudo_secret_b_1,
            "participants": participants
        }

        return operation_index

    # ====

    def multiplication(self, participants, a, b):

        pseudo_secret_a_1 = self.reconstruct_pseudo_secret(participants, a, 1)
        pseudo_secret_b_2 = self.reconstruct_pseudo_secret(participants, b, 2)

        operation_threshold = max(self.t[a] , self.t[b])
        
        operation_index = len(self.t)

        self.t = self.t + [ operation_threshold ]

        self.operation_record[operation_index] = {
            "info (index operation index)": str(a) + " * "+ str(b),
            "operation": "*",
            "operand_a": a,
            "operand_b": b,
            "pseudo_secret_a_1": pseudo_secret_a_1,
            "pseudo_secret_b_2": pseudo_secret_b_2,
            "participants": participants
        }

        return operation_index

# ====

class Client:

    def __init__(self, Participant_Share):
        self.share = Participant_Share

    def sent_share(self):
        return self.share

# ====

if __name__ == '__main__':

    print("\n====\n")

    # n = 1
    n = 30       # n：參與者數量

    # # K：multi-secret list ( K < PRIME )，t：threshold list for each secret (t <= n)。
    # K = [0, 1]
    # t = [0, 0]
    K = [PRIME-1, 7000, 400, 130, 20, 1, 0]
    t = [0, 11, 17, 18, 21, 22, 30]

    dealer = Dealer(n, K, t)

    server, clients = dealer.distribute()

    print("\n====\n")

    # 測試：每筆secret還原狀況

    for i in range(len(K)):

        pool = random.sample(clients, t[i])

        reconstruct = server._reconstruct_Secret(pool, i)
        
        print("Secret:" , K[i] , "Participant_threshold:" , t[i])
        print("Reconstructed secret:", reconstruct)
        print()

    print("\n====\n")

    collection_num = 28

    pool = random.sample(clients, collection_num)

    # 測試：原始 secret 運算狀況

    i , j = 3 , 4

    operation_index_1 = server.addition(pool, i , j)
    reconstruct = server._reconstruct_Secret(pool, operation_index_1)

    print("Secret addition:" , K[i] , "+" , K[j])
    print("Reconstructed secret:", reconstruct)
    print()

    i , j = 3 , 4

    operation_index_2 = server.multiplication(pool, i , j)
    reconstruct = server._reconstruct_Secret(pool, operation_index_2)

    print("Secret multiplication:" , server._reconstruct_Secret(pool, i) , "*" , server._reconstruct_Secret(pool, j))
    print("Reconstructed secret:", reconstruct)
    print()

    print("\n====\n")

    # 測試：運算結果 與 原始 secret 運算狀況

    i , j = operation_index_1 , 4

    operation_index = server.addition(pool, i , j)
    reconstruct = server._reconstruct_Secret(pool, operation_index)

    print("Secret addition:" , server._reconstruct_Secret(pool, i) , "+" , server._reconstruct_Secret(pool, j))
    print("Reconstructed secret:", reconstruct)
    print()

    i , j = operation_index_1 , 4

    operation_index = server.multiplication(pool, i , j)
    reconstruct = server._reconstruct_Secret(pool, operation_index)

    print("Secret multiplication:" , server._reconstruct_Secret(pool, i) , "*" , server._reconstruct_Secret(pool, j))
    print("Reconstructed secret:", reconstruct)
    print()

    i , j = operation_index_1 , operation_index_2

    operation_index = server.addition(pool, i , j)
    reconstruct = server._reconstruct_Secret(pool, operation_index)

    print("Secret addition:" , server._reconstruct_Secret(pool, i) , "+" , server._reconstruct_Secret(pool, j))
    print("Reconstructed secret:", reconstruct)
    print()

    i , j = operation_index_1 , operation_index_2

    operation_index = server.multiplication(pool, i , j)
    reconstruct = server._reconstruct_Secret(pool, operation_index)
    
    print("Secret multiplication:" , server._reconstruct_Secret(pool, i) , "*" , server._reconstruct_Secret(pool, j))
    print("Reconstructed secret:", reconstruct)
    print()

    print("\n====\n")

    server.print_server_record()

    print("\n====\n")