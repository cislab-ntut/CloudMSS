# CloudMSS

MSS_system.py
- import secret_sharing.py：引入 secret sharing 的 generateShares 和 reconstructSecret。
- import multi_secret_sharing.py：引入 multi-secret sharing 的 generate_Participant_Share 和 generate_Public_Shares。

( 角色：Dealer、Client、MSS_system [ 包含 Server_1, Server_2, Randomness_Generator ] )

| **MSS Protocols：本專案提供的 MSS操作 與 實施對象。** |
| :--- |
| MSS.addition()：以 Server_1 主導，兩秘密的加法計算。 |
| MSS.multiplication()：以 Server_1 主導，兩秘密的乘法計算。 |
| MSS.minus()：以 Server_1 主導，兩秘密的減法計算。 |
| MSS.collect_shares()：模擬各方回傳 share 給參與者，當收集到的 share 足夠多，即可恢復出明文。<ul><li>collect_shares_1：主要由 Server_1 傳來的 public shares 和 其他clients 傳來的 participants shares，用來恢復 mask 的明文。</li><li>collect_shares_2：主要由 Server_2 傳來的 public shares 和 其他clients 傳來的 participants shares，用來恢復 masked data 的明文。</li></ul> |
| MSS.reconstruct_MSS_Secret()：以 MSS.collect_shares() 收集各方手上的 shares，回傳由 mask 和 masked data 還原的 secret 明文。 |
| MSS.print_operation_record()：展示 Server_1 儲存的所有計算操作參數。( 這些參數是全部角色都可知的，尤其是 MSS.collect_shares() 會需要各方利用這些參數在本地計算出要傳出的資料 ) |
| MSS.scalar_multiplication()：由 Randomness_Generator 主導，對單一秘密進行純量乘法，可用於 Request Generation 的 查詢資料上傳。 |
| MSS.compare()：以 Server_1 主導，兩秘密的安全比較。 |
| MSS.clear()：清理 Server_1 和 Randomness_Generator 的 operation_record 和 randomness_record。( 結束一個階段的運算後，確認不會在用到過去的計算結果，即可將 record 清理乾淨，節省儲存空間 ) | 

**系統初始化：可參考 application_1.py 的 MSS_system_init()。**
- step 1. 建立 Dealer，設定參與者人數，提供全部的秘密 (請轉換成一個 list)，還有各個秘密對應的門檻值 (請轉換成一個 list)。
- step 2. 建立 Clients，給定 id (不重複的 x 座標)。
- step 3. MSS = dealer.distribute(clients)，開始分發 participant share 給 clients，並且製作 public shares 發送給 雙雲Server，形成 MSS系統。
- step 4. del dealer，完成初始化工作的 Dealer 即可下線 (移除 Dealer)。

----

## MSS_kNN

透過 vscode 開啟資料夾。

執行 application_1.py：模擬 MSS_kNN 運行過程，包括所有 share 計算、client 和 server 的溝通操作。

實驗結果，將會記錄到 application_1__log.txt。

### 實驗結果 與 參數設定

application_1.py：[ 運行順序：__ main __ $\to$ run_code() $\to$ run_epoch() ]


1. Global parameter: 

    - MSS_case = [ (2,2) , (4,2) , (6,2) , (6,4) , (6,6) ]
        
        - (參與者數量, 門檻值)：各 secret 可設定自己的 threshold，此處為求方便評估以固定 threshold 進行實驗。
    
    - B_K = [ 1 ]

        - Basic numbers：一些所有參與者都知道的數字，可將原始功能的常數保留成 MSS系統 的共享，協助常用數字於 MSS計算 的使用 (例如：神經網路的權重 or 其他非機密的參數)，可依開發需求而進行增減。

2. __ main __：本專案提供下列資料集，將依序由左至右運行，可只保留所需的資料集。
   
   - dataName = ['iris' , 'Bankruptcy' , 'banknote' , 'tic-tac-toe' , 'car' , 'breast_cancer']  # sort by total number

        - my_datasets.py：用來載入原始資料集，整理成系統所能正常運行的 dataset。(資料夾 datasets 皆為 UCI Repository 所提供之原始資料，以利當本專案所提供之資料集遺失 or 有更新版本的資料集出現時，能快速地透過本專案進行實驗)

3. run_code()：可設定測試的功能項目、運行次數、資料集分割比例。
    
    - mode = ['knn' , 'dct' , 'MSS_kNN']
        - knn：一般 kNN (無安全性)，用來評估 正確性。
        - dct：一般 DCT (無安全性)，用來評估 正確性。
        - MSS_kNN：Our scheme，用來評估我們的 MSS計算 的工作正確性、耗時。

    -  epoch=10，設定總輪數，計算實驗結果的平均狀況。
    
    -  n_query = 10，設定測試集的資料數量。
        - 每一輪都會重新劃分資料集，但同一輪中的各種 mode 都會以相同的訓練集和測試集，來執行 run_epoch()。
            - test_scale = n_query / len(data)
            - train_X, test_X, train_y, test_y = train_test_split(data, label, test_size = test_scale)
        - 建議：以 my_datasets.py 檢查改動後分類效果是否在可接受範圍。
    
4. run_epoch()：依照上述設定，執行一輪實驗。

    - MSS_kNN()：參數 n_neighbors 可修改，此處設定成定值 5。

----