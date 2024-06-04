# CloudMSS

> This is an experiment based on the research of "Dual-Cloud Multi-Secret Sharing Architecture for Privacy Preserving Persistent Computation".

## Instruction (English)

0. Open this project and get into the `application.py`.

1. In lines 49-51, you can set the `MSS_case`, which assigns the (user, threshold) cases that run in each epoch.

2. In line 222, you can set `epoch`, which gives the repeat rounds and calculate the average results.

3. In line 324, you can set `dataName`, which decides the datasets for the experiment.
 
	- In line 323, you can see the default dataset options provided by this system.

4. Finally, run the code with `python -u "...\CloudMSS\application_1.py"`.

---
### Remark
We are trying to build a user-friendly system with UI. Now, we only have a Chinese draft version in the following link, and the English ver. is still under construction.
> https://github.com/cislab-ntut/Demo_CloudMSS

### Update: Experiments

1. [S-box] We evaluate the representative polynomial for S-box. The result shows roughly 13 seconds to feed an x to complete the evaluation on the share domain. However, it is not independent of the number of secrets. For example, if we directly consider a data block (in 128-bit AES) with 16 Bytes as well as 16 secrets, the average evaluation time can be down to less than 1 second.

---
## 介紹

本研究旨在開發多私密分享的計算工具，以實現高效的多方安全計算技術為目標，從而拓展資源受限設備上的應用場域。

首先，我們以 secret sharing 技術為基底，打造輕量級的密碼工具，形成 multi-secret sharing 的基礎建設。接著，提出雙雲 MSS 保有隱私計算架構，設計持續計算機制，解決客戶端的儲存負擔問題。然後，我們將上述的工作投射到 kNN 的任務上，以雲輔助多私密共享進行多方 kNN 分類服務的相關實驗。最終，完成「實現保有隱私持續計算之雙雲多私密分享架構」的研究。

## 研究論文

- 碩士論文：[實現保有隱私持續計算之雙雲多私密分享架構](./assets/documents/Master%20Thesis%20-%20CloudMSS.pdf)

    - 榮獲 2024 年「賴溪松教授論文獎」碩士組優等獎。

    - 出版於 碩博士論文網 - 國家圖書館：https://hdl.handle.net/11296/msaq3k

    - 投影片：[Master Thesis - CloudMSS ( Oral presentations Slides )](./assets/documents/Master%20Thesis%20-%20CloudMSS%20(%20Oral%20presentations%20Slides%20).pptx)


## 檔案說明

- MSS_system.py：模擬 MSS 系統運作的主要程式。

- secret_sharing.py：提供秘密共享的"分發"與"重構"方法。

- multi_secret_sharing.py：提供多私密分享的"參與者共享"和"公用共享"的生成做法。

- application_1.py：建構 MSS系統 的 kNN 應用。
    
    - 產生 application_1__log.txt 紀錄執行結果。

- my_datasets.py：從 datasets 資料夾 讀取原始資料，並轉換成適合本專案執行的資料格式。

    - 本實驗皆使用 UCI Repository 所提供之原始資料，以利資料集遺失 or 有版本更新時，能快速地進行轉換。

    - 產生 my_datasets__log.txt 紀錄各資料集狀況，用以評估是否適合用以進行實驗。

## MSS 系統

主程式為【MSS_system.py】：模擬各個實體間的互動協議，包括 MSS share 的計算、溝通。

- Dealer 實體：主要負責系統建置，幫助分發 MSS 共享給其他實體。

- Client 實體：用來創建多個用戶的實體，每個用戶只需要維護一份自己的參與者共享。

- MSS_system 實體：包含 Server_1 , Server_2 , Randomness_Generator。

    - Server_1 實體：負責儲存 mask 的公用共享，主導安全計算協議，與 Server_2 形成雙雲的協同運算系統。

    - Server_2 實體：負責儲存 masked data 的公用共享，與 Server_1 搭配為雙雲協同運算系統。

    - Randomness_Generator 實體：負責生成隨機值的 MSS 共享，提供給安全計算協議使用。

### MSS 操作

| 功能 | 用途 | 備註 |
| :---- | :---- | :---- |
| MSS.addition() | 以 Server_1 主導，兩秘密的加法計算。 | |
| MSS.multiplication() | 以 Server_1 主導，兩秘密的乘法計算。 | |
| MSS.minus() | 以 Server_1 主導，兩秘密的減法計算。 | |
| MSS.compare() | 以 Server_1 主導，兩秘密的安全比較。 | |
| MSS.scalar_multiplication() | 由 Randomness_Generator 主導，對 mask 或 masked data 進行純量乘法。 | ( 可用於上傳查詢資料、刷新 MSS 共享。 ) |
| MSS.collect_shares() | 模擬各方傳輸，集中共享。 | <ul><li>collect_shares_1：用來重建 mask 的明文，收集來自 Server_1 和 clients 的共享。</li><li>collect_shares_2：用來重建 masked data 的明文，收集來自 Server_2 和 clients 的共享。</li></ul> |
| MSS.reconstruct_MSS_Secret() | 取得足夠多的共享，還原 secret。 | 以 MSS.collect_shares() 收集各方手上的共享，當數量滿足門檻值，即可還原 mask 和 masked data，用來重建 secret 的明文。 |
| MSS.print_operation_record() | 展示 Server_1 儲存的所有計算操作參數。 | ( 這些參數是全部角色都可知的，尤其是 MSS.collect_shares() 會需要各方利用這些參數在本地計算出要調用的資料。 ) |
| MSS.clear() | 清理 Server_1 和 Randomness_Generator 的 operation_record 和 randomness_record。 | ( 結束一個階段的運算後，確認不會再用到過去的計算結果，即可將 record 清理乾淨，節省儲存空間。 ) |

### 系統初始化

> 在這個項目中【application_1.py】是應用 MSS 系統所開發的範例程式。
> 
> 其中，MSS_system_init() 可供參考。

- step 1. 建立 Dealer。

    - 設定參與者人數，( 整數 )，

    - 加入全部的秘密，( 請轉換成一維列表 list )，

    - 還有各個秘密對應的門檻值，( 大小與秘密相同的一維列表 list )。

- step 2. 建立 Clients。

    - 設定 id，( 不重複的整數，可視為共享的 x 座標 )。

- step 3. 使用【MSS = dealer.distribute(clients)】建置 MSS 系統。

    - 首先，dealer 決定 participant share 給 clients。

    - 接著，dealer 將 secrets 拆分成 protected data ( = mask 和 masked data )。

    - 最後，dealer 建立 MSS 系統，將 protected data 製作成 public shares 發送給兩台 Server。

- step 4. 移除 Dealer，【del dealer】。

    - 完成初始化工作的 Dealer 即可下線。

## 注意事項

1. 本系統具備為每個秘密都設置各自閥值的能力，但為了實驗方便 MSS_kNN 被預設成所有秘密均使用相同的門檻值。 

    - 本實驗提供的 (參與者數量, 門檻值) 案例分析，設定如下：
        
        ~~~
        MSS_case = [ (2,2) , (4,2) , (6,2) , (6,4) , (6,6) ]
        ~~~

2. 本系統提供在初始化階段，將"常用數值"放入 MSS 共享的緩衝區，並且可依開發需求而任意增減。

    - Basic numbers 緩衝區【B_K = []】：將一些原始功能常用的數值，預處理成 MSS 共享，以便 MSS 計算快速調用。

        - 通常是演算法不可缺少的常數，例如：神經網路的初始權重 or 其他非機密的參數。

        - MSS_kNN 的實驗中，我們使用【B_K = [ 1 ]】製作常數 1 的 MSS 共享，協助生成"上傳資料"的 MSS 共享。

## Demo

執行【application_1.py】：使用 MSS 運算，模擬 kNN 的所有過程。

- MSS_kNN：多方保有隱私外包式 k 最鄰近分類。

    - Dual-Cloud Server：共同持有 Dataset 的 MSS 共享，提供 kNN 分類服務。

    - Clients：每人持有一份參與者共享，可以獨自上傳 query data，需要有夠多同伴的協助才能得到分類結果。

    - 單一的雲服務器，無法破解 MSS 計算系統，使用「非共謀 ( non-colluding )」的雙雲服務器，形成安全的計算服務。

    - 使用 MSS 達成多方資料之間的保有隱私共同計算，避免使用一對一兩方安全計算，形成如完全連通圖的複雜溝通網路。

    - 用戶們可以控管資料的使用權，以分散式管理的做法，避免信任度不足的攻擊者，擁有獲得服務的權限。

- 本專案之目的為收集相關實驗數據，著重於評估基於我們的工作所建立的安全服務，其對不同的設定配置所產生的影響。

    - 實驗參數配置方案如下所示：(參與者數量, 門檻值)
        
        ~~~
        MSS_case = [ (2,2) , (4,2) , (6,2) , (6,4) , (6,6) ]
        ~~~

- 執行結果，將會記錄到 application_1__log.txt。

    - 以 Report_and_Problem 資料夾，整合紀錄所有的實驗數據。

### 運行流程、控制參數

application_1.py 的運行順序：\_\_main\_\_ -> run_code() -> run_epoch()。

1. \_\_main\_\_：選擇資料集，由左至右運行。
    
    - dataName = ['iris' , 'Bankruptcy' , 'banknote' , 'tic-tac-toe' , 'car' , 'breast_cancer']

2. run_code()：設定測試的功能項目、運行次數、資料集分割比例。
    - mode = [ 'knn' , 'dct' , 'MSS_kNN' ]
        
        - knn：測量 ( 正確性基準 )，一般 kNN，無安全性。
        - dct：測量 ( 正確性基準 )，一般 DCT，無安全性。
        - MSS_kNN：測量 ( 正確性、耗時 )。

    -  epoch = 10，設定總輪數，計算實驗結果的平均狀況。
    
    -  n_query = 10，設定測試集的資料數量。
        
        - 每一輪都會重新劃分資料集，但同一輪中的各種 mode 都會以相同的訓練集和測試集，來執行 run_epoch()。
            
            ~~~
            test_scale = n_query / len(data)
            
            train_X, test_X, train_y, test_y = train_test_split(data, label, test_size = test_scale)
            ~~~
      
3. run_epoch()：依照上述設定，執行實驗。

    - MSS_kNN()：參數 n_neighbors 可修改 k 值，此實驗預設值為 5。

### 實驗結果

~~~

===========

資料集:iris
Instances: 150 , Attributes: 4 , Class: 3 => Total: 600

Epoch:  10  => (Train:  140 , Test:  10 , Test/All:  0.06667 )
Mode: knn      	 Case: None 	 正確率: 94.0 % 	 耗時: 0.001341390609741211
Mode: dct      	 Case: None 	 正確率: 93.0 % 	 耗時: 0.000497746467590332
Mode: MSS_kNN  	 Case: (2, 2) 	 正確率: 93.0 % 	 耗時: 52.746141052246095
Mode: MSS_kNN  	 Case: (4, 2) 	 正確率: 93.0 % 	 耗時: 106.25199618339539
Mode: MSS_kNN  	 Case: (6, 2) 	 正確率: 93.0 % 	 耗時: 196.11596415042877
Mode: MSS_kNN  	 Case: (6, 4) 	 正確率: 93.0 % 	 耗時: 192.2788189649582
Mode: MSS_kNN  	 Case: (6, 6) 	 正確率: 93.0 % 	 耗時: 187.6793925046921

===========

資料集:Bankruptcy
Instances: 250 , Attributes: 6 , Class: 2 => Total: 1500

Epoch:  10  => (Train:  240 , Test:  10 , Test/All:  0.04 )
Mode: knn      	 Case: None 	 正確率: 98.0 % 	 耗時: 0.0016557693481445313
Mode: dct      	 Case: None 	 正確率: 100.0 % 	 耗時: 0.0007044076919555664
Mode: MSS_kNN  	 Case: (2, 2) 	 正確率: 98.0 % 	 耗時: 137.6401345729828
Mode: MSS_kNN  	 Case: (4, 2) 	 正確率: 98.0 % 	 耗時: 274.21283826828005
Mode: MSS_kNN  	 Case: (6, 2) 	 正確率: 98.0 % 	 耗時: 507.512745642662
Mode: MSS_kNN  	 Case: (6, 4) 	 正確率: 98.0 % 	 耗時: 497.35079782009126
Mode: MSS_kNN  	 Case: (6, 6) 	 正確率: 98.0 % 	 耗時: 484.4957030773163

===========

資料集:banknote
Instances: 1372 , Attributes: 4 , Class: 2 => Total: 5488

Epoch:  10  => (Train:  1362 , Test:  10 , Test/All:  0.00729 )
Mode: knn      	 Case: None 	 正確率: 100.0 % 	 耗時: 0.00284273624420166
Mode: dct      	 Case: None 	 正確率: 98.0 % 	 耗時: 0.0023366689682006838
Mode: MSS_kNN  	 Case: (2, 2) 	 正確率: 100.0 % 	 耗時: 547.202475643158
Mode: MSS_kNN  	 Case: (4, 2) 	 正確率: 100.0 % 	 耗時: 1067.6747081041335
Mode: MSS_kNN  	 Case: (6, 2) 	 正確率: 100.0 % 	 耗時: 1947.8911508321762
Mode: MSS_kNN  	 Case: (6, 4) 	 正確率: 100.0 % 	 耗時: 1913.679037833214
Mode: MSS_kNN  	 Case: (6, 6) 	 正確率: 100.0 % 	 耗時: 1861.3711141347885

===========

資料集:tic-tac-toe
Instances: 958 , Attributes: 9 , Class: 2 => Total: 8622

Epoch:  10  => (Train:  948 , Test:  10 , Test/All:  0.01044 )
Mode: knn      	 Case: None 	 正確率: 100.0 % 	 耗時: 0.0015516042709350585
Mode: dct      	 Case: None 	 正確率: 95.0 % 	 耗時: 0.0013024568557739257
Mode: MSS_kNN  	 Case: (2, 2) 	 正確率: 100.0 % 	 耗時: 887.9076585292817
Mode: MSS_kNN  	 Case: (4, 2) 	 正確率: 100.0 % 	 耗時: 1707.864125418663
Mode: MSS_kNN  	 Case: (6, 2) 	 正確率: 100.0 % 	 耗時: 3091.6753043174745
Mode: MSS_kNN  	 Case: (6, 4) 	 正確率: 100.0 % 	 耗時: 3026.5948858737947
Mode: MSS_kNN  	 Case: (6, 6) 	 正確率: 100.0 % 	 耗時: 2952.101217055321

===========

資料集:car
Instances: 1728 , Attributes: 6 , Class: 4 => Total: 10368

Epoch:  10  => (Train:  1718 , Test:  10 , Test/All:  0.00579 )
Mode: knn      	 Case: None 	 正確率: 98.0 % 	 耗時: 0.0018868446350097656
Mode: dct      	 Case: None 	 正確率: 99.0 % 	 耗時: 0.001337122917175293
Mode: MSS_kNN  	 Case: (2, 2) 	 正確率: 99.0 % 	 耗時: 1091.932049202919
Mode: MSS_kNN  	 Case: (4, 2) 	 正確率: 99.0 % 	 耗時: 2092.710607767105
Mode: MSS_kNN  	 Case: (6, 2) 	 正確率: 99.0 % 	 耗時: 3744.5810544013975
Mode: MSS_kNN  	 Case: (6, 4) 	 正確率: 99.0 % 	 耗時: 3675.813367462158
Mode: MSS_kNN  	 Case: (6, 6) 	 正確率: 99.0 % 	 耗時: 3583.021878528595

===========

資料集:breast_cancer
Instances: 569 , Attributes: 30 , Class: 2 => Total: 17070

Epoch:  10  => (Train:  559 , Test:  10 , Test/All:  0.01757 )
Mode: knn      	 Case: None 	 正確率: 93.0 % 	 耗時: 0.001808309555053711
Mode: dct      	 Case: None 	 正確率: 95.0 % 	 耗時: 0.006652235984802246
Mode: MSS_kNN  	 Case: (2, 2) 	 正確率: 94.0 % 	 耗時: 2067.7960641622544
Mode: MSS_kNN  	 Case: (4, 2) 	 正確率: 94.0 % 	 耗時: 3688.9246839284897
Mode: MSS_kNN  	 Case: (6, 2) 	 正確率: 94.0 % 	 耗時: 6411.448307204247
Mode: MSS_kNN  	 Case: (6, 4) 	 正確率: 94.0 % 	 耗時: 6304.07366309166
Mode: MSS_kNN  	 Case: (6, 6) 	 正確率: 94.0 % 	 耗時: 6148.930614113808

===========

~~~

## 開發紀錄

0. Design goal
   
    (1) 以 MSS 降低 client 所需持有或處理的共享。
   
    (2) 研究基於 MSS 的多方高效安全計算方式。

1. Single Server version
    
    - done：
        
        (1) 實作 MSS 工具：[ An Efficient Verifiable Threshold Multi-Secret Sharing Scheme With Different Stages ]。
        
        (2) 建構高效安全計算方法：[ Secrecy Computation without Changing Polynomial Degree in Shamir’s (K, N) Secret Sharing Scheme ]。
  
    - problem：
    
        (1) 直接傳輸原始的 Client share 會有被竊聽的危險。
  
        (2) 只要 Server 拿到足夠多的 Client share，即具備破解 secret 的能力。
    
        (3) 當兩個 secret 的數值相同，會生成相同的 public shares，即可在不解密的情況下知曉明文等價的關西。
        
    - solution idea：
        
        (1) 增加 share randomization 保有傳輸資料的隱私。
        
        (2) 考慮改成由 client 蒐集 share 來解密。

        (3) 將 secret 拆成 mask 和 masked data 分別交給兩個不可串通的 Server 來管理。

2. Dual Server version

    - goal：改善 MSS 系統的安全性。
        
    - done：
        
        (1) 增加 Randomness_Generator 以 Beaver triple 的做法，隨機化多方共享。 
        
        (2) 在安全計算的參與者中，隨機選擇任一 client 收集各方共享，解開所需資料，完成安全計算。
        
        (3) 以 dual server 分別儲存 mask / masked data。
  
    - problem：考慮將安全計算工作指派給 Server 處理，探討增加 dual server 和 RG 的方式，是否能安全地幫助 client 執行計算工作。

3. Server Compuation version

    - goal：由 Server 主導安全計算工作。
        
    - done：

        - 確保使用 server 計算的安全性。

            (1) dual server 讓 public shares 不集中在單一角色中。
        
            (2) randomization 使 share 只能用於本輪計算工作。
        
        - 建構持續計算機制，避免讓參與者維護這些計算結果共享。
            
            ~~~
            將"計算過程"以參數方式儲存到 Record，並在調用時重建計算結果。

            - randomness_record：紀錄隨機化參數。

            - operation_record：紀錄運算參數。
            ~~~

    - problem：增加 secure comparison 的功能。

4. Comparison protocol with random_calculate_share

    - goal：以 2-out-of-2 的 Secure comparison 為基底，建構多方安全比較協議。
        
    - done：採用隨機計算得出亂數的做法，依靠 MSS_multiplication 和 MSS_addition 產生一個隨機數的 MSS share。
        
    - problem：這種隨機數的產生做法，需要進行多次的 MSS 計算。我們需要一種更省時省力的做法，將數值安全地轉換成新的 MSS share。【不受限於使用原始 secret 的計算，而能將一些新數字上傳成 MSS 共享。】

5. Data uploading by scalar_multiplication

    - goal：改善使用 MSS 計算生成新資料的 MSS 共享。
        
    - done：透過 Randomness Generator 做出新資料的安全上傳機制。
        
    - problem：嘗試這些功能的平行化效果。

6. Testing parallel 

    - goal：嘗試這些功能的平行化能力，探討 batch computing 的效果。
        
    - done：由於我們的情境是多方計算，所以嘗試以平行化寫法，將 share 的計算工作分散到 n 個 process 或 thread 之中。
        
    - problem：實際上，我們還是只用一台電腦模擬所有計算工作，增加平行化會產生更多建立 process 的步驟，而平行化本身看不出來有效能上的提升，因此反而會降低計算效率。
        
        - 因此，我們後續的開發工作將從 "create Data uploading by scalar_multiplication" 的版本出發。
        
        - 本次 commit 用來記錄平行化相關測試之程式開發，視工作目標與考慮情境的不同，不能單以此平行化測試完全否定 MSS 平行化的可能性。
    
7. Naive application version

    - goal：以 MSS_system 為基底，開發 MSS 應用。
        
    - done：建構 MSS_kNN ( application_1.py ) 用來評估多方保有隱私計算的系統運作。
        
    - problem：發生 Memory Error ( 記憶體不足 ) 的錯誤。此問題來自大型資料集帶來的大量計算工作，產生大量未釋放的暫存計算結果 ( operation_record )。

8. Fulfilled application version

    - goal：將不會再使用到的計算資料從 record 移除，減少不必要的 memory 開銷。
        
    - done：

        (1) 每一筆 query data 的查詢，都是對整個 dataset 的所有資料重新進行距離計算。分類出一筆 query data 後，舊的計算結果不會再影響到下一筆資料的分類過程，所以可以透過 clear reocrd 增加記憶體空間。
        
        (2) 成功，完成對大型資料集的 MSS_kNN 進行實驗。
        
    - problem：

        (1) 利用 secret sharing 的 secure comparison 調適成 MSS secure comparison 出錯。
        
        (2) Randomness Generator 可能有潛在安全漏洞。

9. Fail comparison version

    - goal：嘗試 MSS secure comparison 的簡化做法。
       
        - 上一版做法：以 MSS 計算，置換 secret sharing 的 Secure comparison 計算。
       
        - 簡化做法：參考 MSS addition，將兩個 secret 轉換成相同的 mask 下，再用一般的 Secure comparison 進行比較。

    - problem：
        
        - 失敗，secure comparison 需要設置參數 $\ell$ 協助讓差值保持正數 ( 避免 modulo 影響結果 )，再進行比較。
        
        - 然而，對於上述做法所產生的 masked data $( [r_1 r_2 \alpha \beta x],  [r_1 r_2 \alpha \beta y] )$，在共享刷新之後，難以確定 masked data 的資料範疇，所以無法給出適合的預設參數 $\ell$。
        
        - 因此，還是使用 MSS 計算 進行 Secure Comparison，方可確保計算過程不會影響原始 secret 的數值範圍，以利我們透過原始資料評估可用的預設參數 $\ell$。

10. Final version

    - goal：解決 Randomness Generator 的安全漏洞。
        
    - done：改成使用 server 恢復 d, e 的明文。避免使用 RG 還原 $d = x - a$，造成 RG 知道 a，也得以破解秘密 x 內容的問題。
        
