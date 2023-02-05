from ctypes import sizeof
import random 
import numpy as np
import pandas as pd
import time
import os

from sklearn import datasets
from sklearn import tree
from sklearn import neighbors
from sklearn.model_selection import train_test_split

from my_datasets import loading_datasets

from MSS_system import *

# =======

# Global Parameters

# Mersenne Prime 4th(7) 5th(13) 6th(17) 7th(19) = 127, 8191, 131071, 524287
PRIME = 2**19 - 1

# L的最小值，必需比資料最大值(包括計算後)來的大。
L_Min = 3000

random_size = 100

# ====

# # u：參與者數量。
# u = 2
# u = 5
u = 10
# u = 100

# # T：為求便於計算各項 secret 的 threshold 固定成 T。
# T = 2
# T = 3
T = 5
# T = 20

# # Basic numbers：協助運算的已知 secret。
B_K = [ PRIME-1 , 1 ]
B_t = [ 1 , 1 ]
b_k = len(B_K)

# =======

def MSS_system_init(train_X, train_y):
    
    n_row = train_X.shape[0]
    n_column = train_X.shape[1]

    # # K：multi-secret list ( K < PRIME )，PRIME = 524287。
    # K = [0, 1, 130, 20, 1500, 700, 400, 2100, 2800, 1300]
    K = list( train_X.flatten('C') )    # order = 'C'：以列為主。

    # # t：threshold list for each secret (1 <= t <= n)，threshold 可非固定。
    t = [T] * len(K)                    # 此處為求便於計算 threshold 固定，size 與 secret 一樣多。

    # 加入 Basic numbers：協助運算的已知 secret。
    K = B_K + K
    t = B_t + t

    # Dealer 收到 整合資料集。
    dealer = Dealer(u, K, t, train_y)

    # User 收到 獨立id = x座標。
    clients = []        
    for i in range(u):
        clients.append(Client(i))

    # 開始分發 User share & 製作 雙雲Server 的 public share。
    MSS = dealer.distribute(clients)

    del dealer

    return MSS, clients, n_row, n_column

def MSS_kNN(MSS, participant, n_row, n_column, test_X, n_neighbors):
    
    result = []

    labels = MSS.sent_labels()

    for i in range(test_X.shape[0]):

        print("\n-query instance:", i, end="")

        # STEP1 client 將 每一筆 query_instance 製成 share。
        query_share_index = []
        
        for j in range(test_X.shape[1]):
            query_share_index.append(MSS.scalar_multiplication(participant, 1 , test_X[i][j]))

        # STEP2 計算 dataset instance 與 此 query instance 的距離。
        distance = []

        for r in range(n_row):
            deltaSum = 0

            for c in range(n_column):

                d_attribute_index = b_k + (r * n_column) + c
                q_attribute_index = query_share_index[c]

                # STEP2-1 每個 attribute 比較大小：dataset_attribute > query_attribute = 1。
                comp = MSS.compare(participant, d_attribute_index , q_attribute_index)

                # STEP2-2 計算 difference
                difference = 0
                if comp == 1:
                    difference = MSS.minus(participant, d_attribute_index , q_attribute_index)
                else:
                    difference = MSS.minus(participant, q_attribute_index , d_attribute_index)

                # STEP3 計算距離
                delta = MSS.reconstruct_Secret(participant, difference)

                # print("c:", c , "compare:", comp , "delta:", delta)

                deltaSum += delta**2

            # os.system("pause")

            deltaSum = deltaSum**0.5

            distance.append(deltaSum)

        result += knn_classifier(distance, labels, n_neighbors)
        
    return result

def knn_classifier(distance, labels, n_neighbors):          # distance = 每筆 train instance 跟 某個 query instance 的距離

    result = []

    DAL = []                                        # DAL = DistanceAndLabel
    for i in range(len(distance)):
        d = distance[i]
        l = labels[i]
        DAL.append([ d , l ])
    
    SDAL = sorted( DAL , key=(lambda x : x[0]) )    # SDAL = Sorted_DAL  (按距離由小到大排序)
    
    predict_label = {}
    neighbor_distance = SDAL[0][0]
    neighbor_count = 0
    for i in range(len(SDAL)):
        d = SDAL[i][0]
        l = SDAL[i][1]

        if d > neighbor_distance:
            if neighbor_count > n_neighbors:
                break
            else:
                neighbor_distance = d

        if d <= neighbor_distance:
            predict_label[l] = predict_label.get(l, 0) + 1
            neighbor_count += 1

    SPL = sorted( predict_label.items() , key=(lambda x:x[1]), reverse=True )

    result.append(SPL[0][0])
    
    return result

def acc_evaluate(result, test_y):
    
    correct_rate = 0

    if(len(result) != 0):
        incorrect = 0
        for j in range(len(result)):
            if( result[j] != test_y[j] ):
                incorrect = incorrect + 1
        
        correct_rate = ( len(result) - incorrect ) / len(result) * 100
    
    return correct_rate

# =======

def run_code(dataset):
    
    mode = ['knn' , 'MSS_kNN' , 'dct']
    # mode = ['knn']

    data = dataset.data
    label = dataset.target
    NUM_CLASS = dataset.NUM_CLASS
    dataName = dataset.dataName

    # print('\n====\n')
    # print('資料集:{}'.format(dataName))
    # print('Instances: {} , Attributes: {} , Class: {}' .format( len(label) , len(data[0]) , NUM_CLASS ) )

    print('\n====\n', file=open('application_1__log.txt', 'a+'))
    print('資料集:{}'.format(dataName), file=open('application_1__log.txt', 'a+'))
    print('Instances: {} , Attributes: {} , Class: {}' .format( len(label) , len(data[0]) , NUM_CLASS ) , file=open('application_1__log.txt', 'a+'))


    # epoch=1
    # epoch=3
    epoch=5
    # epoch=10

    # 每一種 mode 存放一組資料 [ mode_name, total_accuracy = 0 , total_time_cost = 0 ]
    mode_record = []

    if(isinstance(mode, list)):
        for i in range(len(mode)):
            mode_i = mode[i]
            mode_record.append([ mode_i , 0 , 0 ])

    else:        
        mode_record = [ mode , 0 , 0 ]
    
    for e in range(epoch):
        
        print('\nepoch e: ', e , end = ' ')
        
        # 切分訓練與測試資料
        test_num = 10 / len(data)
        train_X, test_X, train_y, test_y = train_test_split(data, label, test_size = test_num)
        # train_X, test_X, train_y, test_y = train_test_split(data, label, test_size = 0.03)

        # 建立 MSS 系統
        MSS, clients, n_row, n_column = MSS_system_init(train_X, train_y)

        # 隨機參與者：人數 = T。
        pool = random.sample(clients, T)

        # ===========

        if(isinstance(mode, list)):
            for i in range(len(mode)):
                mode_i = mode[i]
                result , time_cost = run_epoch(mode_i, MSS, pool, n_row, n_column, train_X , train_y, test_X)
                accuracy = acc_evaluate(result, test_y)
                
                mode_record[i][1] += accuracy
                mode_record[i][2] += time_cost
        else:
            result , time_cost = run_epoch(mode, MSS, pool, n_row, n_column, train_X , train_y, test_X)
            accuracy = acc_evaluate(result, test_y)
            
            mode_record[1] += accuracy
            mode_record[2] += time_cost

        del MSS
        del clients
    
    print()

    print('\nEpoch: ', epoch, ' => (Train: ', len(train_X), ', Test: ', len(test_X), ', Test/All: ', round(len(test_X)/len(data), 5), ")")
    print('\nEpoch: ', epoch, ' => (Train: ', len(train_X), ', Test: ', len(test_X), ', Test/All: ', round(len(test_X)/len(data), 5), ")", file=open('application_1__log.txt', 'a+'))

    if(isinstance(mode, list)):
        for i in range(len(mode)):
            print('Mode:' , mode_record[i][0] , ' 正確率 ' , mode_record[i][1] / epoch , '%' , ' 耗時 ' , mode_record[i][2] / epoch)
            print('Mode:' , mode_record[i][0] , ' 正確率 ' , mode_record[i][1] / epoch , '%' , ' 耗時 ' , mode_record[i][2] / epoch , file=open('application_1__log.txt', 'a+'))
    else:
        print('Mode:' , mode_record[0] , ' 正確率 ' , mode_record[1] / epoch , '%' , ' 耗時 ' , mode_record[2] / epoch)
        print('Mode:' , mode_record[0] , ' 正確率 ' , mode_record[1] / epoch , '%' , ' 耗時 ' , mode_record[2] / epoch , file=open('application_1__log.txt', 'a+'))

    return 0

def run_epoch(mode, MSS, participant, n_row, n_column, train_X , train_y, test_X):

    time1 = time.time()

    result = []

    if(mode == 'MSS_kNN'):                                  # MSS_kNN：正確性、耗時
        result = MSS_kNN(MSS, participant, n_row, n_column, test_X, 5)
    elif(mode == 'knn'):                                    # knn：正確性
        classifier = neighbors.KNeighborsClassifier(n_neighbors = 5, algorithm = 'brute')
        classifier = classifier.fit(train_X, train_y)
        result = classifier.predict(test_X)
    elif(mode == 'dct'):                                    # dct：正確性
        classifier = tree.DecisionTreeClassifier()
        classifier = classifier.fit(train_X, train_y)
        result = classifier.predict(test_X)
    
    time2 = time.time()

    time_cost = time2 - time1

    if(mode == 'MSS_kNN'):
        print()
        print("MSS_kNN time_cost:", time_cost)

    return result, time_cost

if __name__ == '__main__':

    print('', file=open('application_1__log.txt', 'w'))

    # dataName = ['iris' , 'wine' , 'breast_cancer' , 'abalone' , 'digits' , 'nursery' , 'mushroom' , 'Chess (King-Rook vs. King)']
    # dataName = ['iris' , 'breast_cancer' , 'digits' , 'mushroom' , 'nursery']
    # dataName = ['iris' , 'breast_cancer']
    dataName = ['iris']

    if(isinstance(dataName, list)):
        while(len(dataName) > 0):
            dataName_i = dataName.pop(0)
            dataset = loading_datasets(dataName_i)
            run_code(dataset)
    else:
        dataset = loading_datasets(dataName)
        run_code(dataset)

    print('\n====\n')
    print('\n====\n', file=open('application_1__log.txt', 'a+'))

# =======