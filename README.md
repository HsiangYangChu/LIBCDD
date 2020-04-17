# ConceptDriftDetectionLib

### 2020年2月24日之前

#### Reading:

* Lujie 2018 Learning under Concept Drift A Review

* Learning from Time-Changing Data with Adaptive Windowing (**ADWIN_2007**)

* Learning with Drift Detection (**DDM_2004**)

* Exponentially Weighted Moving Average Charts for Detecting Concept Drift (**ECDD_2012**)

* Early Drift Detection Method (**EDDM_2006**)

* Concept Drift Detection Based on Equal Density Estimation (**EDE_2016**)

* Fuzzy Time Windowing for Gradual Concept Drift Adaptation (**FW-DDM_2017**)

* Learning with Local Drift Detection (**LLDD_2006**)

* Detecting Concept Drift Using Statistical Testing (**STEPD_2007**)

* 浙江大学版《概率论与数理统计》

* 感知机原理

* 朴素贝叶斯方法原理

* 决策树原理

* 周志华《机器学习》

#### Done:

* ***ECDD***

#### Nodes:

* **FW-DDM**中的方法交代的不清楚，![](https://www.zhihu.com/equation?tex=u_{old}) 和 ![](https://www.zhihu.com/equation?tex=u_{new}) 方法没有讲清，而且提供的源码简直驴唇不对马嘴，差评

* ***ECDD***的方法较好，而且背后的数学理论性较强，点赞

---

### 2020年2月24日

#### Reading:

* Detecting Concept Drift Using Statistical Testing (**STEPD_2007**)

#### Done:

* ***STEPD***

#### Nodes:

* **STEPD_2007**中的假设检验![](https://www.zhihu.com/equation?tex=H_1)是不对的，实验结果不理想

---

#### 2020年2月25日

#### Reading:

* Online and Non-Parametric Drift Detection Methods Based on Hoeffding’s Bounds (**HDDM_2014**)

#### Done:

* ***HDDM_A***

* ***HDDM_W***

#### Nodes:

* 总算是把**FW-ECDD**给实现了，但是对文中的方法还是存疑

* Reading了四页**HDDM_2014**，将github上大牛的方法copy下来了

---

### 2020年2月27日

#### Reading:

* Online and Non-Parametric Drift Detection Methods Based on Hoeffding’s Bounds (**HDDM_2014**)

* 霍夫丁不等式

* 切比雪夫不等式

* 马尔科夫不等式

#### Done:

* HDDM_A（自己敲了一遍，并修改了一些错误）

#### Nodes:

* 发现了HDDM_A中的一些与论文不一致的地方，参数和公式推导的错误

---

### 2020年2月28日

#### Reading:

* Online and Non-Parametric Drift Detection Methods Based on Hoeffding’s Bounds (**HDDM_2014**)

#### Done:

* **HDDM_W**

#### Nodes:

* 首先，感觉**TKDE**(IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING)不愧为CCF推荐A类期刊，确实水平比较高
理论性比较强，排版也很严谨，没有发现typo

* 我发现了一处不理解的地方（有可能是我没理解，也有可能是作者的失误，但是总体上对结果影响不大）

![img](./images/2020_02_28.jpg)

作者说上述左右两边是等价的，但是仔细推导发现还是有一些不同

---

### 2020年2月29日

#### Reading:

* Online and Non-Parametric Drift Detection Methods Based on Hoeffding’s Bounds (**HDDM_2014**)

* Dynamic extreme learning machine for data stream classification (**DELM_2017**)

* Extreme learning machine (**ELM**)原理

* **OS-ELM**原理

#### Done:

* ***DELM***（在网上找到份源码，但还没有统一接口）

#### Nodes:

* **DELM_2017**中有很多公式，这些公式大体上能看懂，深究还是有点迷，而且文章的编排方式比较乱，但是还是有不错的实际意义的

* **DELM_2017**是一个专门的DM，不具有泛化能力，他是在***ELM***和***OS-ELM***的基础上发展而来的，对于特定的情况优势明显

---

### 2020年3月1日

#### Reading:

* Concept drift detection via competence models (**CM_2014**)

* Dynamic extreme learning machine for data stream classification (**DELM_2017**)

* Concept Drift Detection Based on Equal Density Estimation (**EDE_2016**)

#### Done:

* ***ELM***

* ***CusumDM***

#### Nodes:

* 在Done最后一个算法***CusumDM***之后，就算把第一大类也是主流的一类error-base方法结束了，接下来就是date-base了

* 在date-base中我首先回顾了**EDE_2016**，作者在文中仅仅提供了一个开放的思想，具体的实现并没有细说，但是基本思想还是不错的，在距离
函数方面还有很多讨论的空间

* 读了四页**CM_2014**，刚好把related work看完，发现concept drift领域可以说是百花齐放，各种方法各有优点，但是主流的就那几个

---

### 2020年3月2日

#### Reading:

* Concept drift detection via competence models (**CM_2014**)

* RDDM: Reactive Drift Detection Method (**RDDM_2017**)

#### Done:

* ***RDDM***

* ***GeometricMovingAverageDM***

#### Nodes:

* **CM_2014**这篇文章里面太多数学的东西，一时还不能完全看懂，但是里面关于集合论的一些定义、定理及其证明还是很有意思的

* **RDDM_2017**这篇文章提出了一个***RDDM***方法，他是对***DDM***的一个改进，因为***DDM***不适合drift比较慢和concept数量比较大的情况，***RDDM***有针对性行的在这方面
做了很多提高，但是它也有一个很大的弱点，那就是需要用户定义的变量太多

---

### 2020年3月3日

#### Reading:

* Concept drift detection via competence models (**CM_2014**)

* Permutation test（置换检验）

* Kolmogorov–Smirnov test（K-S检验）

#### Done:

* ***Permutation test***

#### Nodes:

* 又看了几个小时的**CM_2014**，现在发现慢慢的可以看得懂了，并且发现自己在放松的时刻效率会比较高

* 学习了并实现了***Permutation test***，以前老是被**Permutation test**卡到，今天终于知道这个什么东西了，很好用

* **Kolmogorov–Smirnov test**，太深邃了没看懂，不过知道这个干什么的了

---

### 2020年3月4日

#### Reading:

* Concept drift detection via competence models (**CM_2014**)

* 周志华《机器学习》之聚类算法

#### Done:

* ***Permutation test***

#### Nodes:

* 今天终于把**CM_2014**给看完了，虽然有些地方还不能理解，但是大体意思还是明白的

---

### 2020年3月5日

#### Reading:

* A concept drift-tolerant case-base editing technique (**CM_2016**)

* 周志华《机器学习》之降维算法

#### Done:



#### Nodes:

* 看完了**CM_2016**的主要思想，实验还没看，这篇与**CM_2014**都是来自于Jie Lu(IEEE Fellow)团队，文中提出了一个新的case-base editing approach，他分为三步：
    
    * 1、根据**CM_2014**提出的方法进行detection

    * 2、Competence Enforcement，大致目的就是去除noise，使用的方法为NEFCS(Noise-enhanced fast context switching)，它也包含三个部分，分别为M-BBNR(Modified blame-based noise reduction)、Context switching和Update competence model

    * 3、Stepwise redundancy removal，目的就是去除冗余，提高algorithm的effectiveness
    
--- 

### 2020年3月7日

#### Reading:

* An Information-Theoretic Approach to Detecting Changes in Multi-Dimensional Data Streams (**ITA_2006**)

* KD-Tree原理

* Detecting Change in Data Streams

#### Done:



#### Nodes:

* 看完了**ITA_2006**，通过这篇文章的Reading，意识到，对与前段时间研究的error-base方法，现在的这一类方法因为无法确定数据的分布，所以无法假设数据的分布，只能采取nonparametric tests
    
    * Test Statistics Calculation: KL-distance
    
    * Statistical Bounds: BootStrap method
    
    * 优点：思路清晰，思路简单，可以用于高位数据，可解释，高效率

* Detecting Change in Data Streams这篇文章太深邃的，没看懂，但是发现之前Reading的**CM_2014**和**CM_2016**中部分借鉴了文中的方法

* KD-Tree是BST的高位拓展，一般用来实现KNN


--- 

### 2020年3月10日

#### Reading:

* Prototype-based Learning on Concept-drifting Data Streams (**SyncStream**)

* A PCA-Based Change Detection Framework for Multidimensional Data Streams (**PCA-CD**)

#### Done:



#### Nodes:

* **SyncStream**是一篇很不错的论文，以下是大致总结

    * 1、该方法讲gradual drift和abrupt drift分开处理，前者使用clustering以缓慢适应、后者使用PCA（计算vector夹角）或者Statistical Test（Wilcoxon）
    
    * 2、提出了一个叫做P-Tree的数据结构，该数据结构分两层，第一层存储代表代表当前concept的prototypes、第二层是历史concepts
    
    * 3、采用最近邻学习法，保留原始结构，定期更新原形，显示P-Tree空间

--- 

### 2020年3月10日

#### Reading:

* A PCA-Based Change Detection Framework for Multidimensional Data Streams (**PCA-CD**)

* A pdf-Free Change Detection Test Based on Density Difference Estimation (**LSDD-CDT**)

#### Done:

* ***SyncStream***(Java)

#### Nodes:

* **PCA-CD**是一篇很不错的论文，以下是大致总结

    * Estimating Density Functions: 在数据维度比较的时候使用Histograms，低维使用KDE-Track（精度和复杂度的权衡）
    
    * Divergence Metrics: MKL、A and LLH
    
    * Hypothesis Test: Page-Hinley
    
    * 每隔step时间进行一下测试
    
    * Windows: Reference Window & Current Window (sliding)
    
    * PCA
    
    * Dynamic Threshold Settings
   
--- 

### 2020年3月12日

#### Reading:

* A pdf-Free Change Detection Test Based on Density Difference Estimation (**LSDD-CDT**)

* Reservoir Sampling Mechanism

* BootStrap Method

* An Incremental Change Detection Test Based on Density Difference Estimation (**LSDD-INC**)

#### Done:

* ***PCA-CD***(C++)

#### Nodes:

* Summary of **LSDD-CDT**

    * BootStrap Method
    
    * Using Reservoir Sampling for reference window when it is not in the warning condition or has a false alarm
    
    * Three-level Threshold Mechanism: To be more sensitive to changes(i.e. keep low FNs), yet remaining the same FP rate
    
    * Drawbacks: High computational complexity(O(n^2)) and not feasible in datastreams
    
--- 

### 2020年3月12日~15日

#### Reading:

* An Incremental Change Detection Test Based on Density Difference Estimation (**LSDD-INC**)

* Regional Concept Drift Detection and Density Synchronized Drift Adaptation (**LDD-DSDA**)

#### Done:

* ***LDD-DSDA***(matlab)

#### Nodes:

---

### 2020年3月15日~21日

#### Reading:

* Concept Drift Detection for Streaming Data (**LRF_2015**)

* Just-in-Time Adaptive Classifiers—Part I: Detecting Nonstationary Changes (**JIT1_2007**)

* Hierarchical Change-Detection Tests (**HCDTs_2016**)

* A Lightweight Concept Drift Detection Ensemble (**DDE_2015**)

#### Done:

* ***DDE***

#### Nodes:

* Summary of **LRF_2015**

    * It's a parallel multiple hypothesis test, which has four statistics, namely, tpr, tnr, ppv, npv
    
    * It's computational complexity is O(1), whereas offline computations(e.g. BoundTable) need to be done

* Summary of **JIT1_2007**

    * extended CUSUM & CI-CUSUM(PCA)
    
    * Provide 4 configurations
    
* Summary of **HCDTs_2016**

    * A Hierarchical Structure was provided, which has two layers, namely, Detection Layer and Validation Layer

* Summary of **DDE_2015**

    * A Parallel Structure was provided, a combination of three detections, e.g., {HDDMA, HDDMW, DDM} 
    
--- 

### 2020年3月22日

#### Reading:

* Reacting to Different Types of Concept Drift: The Accuracy Updated Ensemble Algorithm (**AUE2_2013**)

* Friedman test & Bonferroni-Dunn test

#### Done:

* ***DDE***

#### Nodes:

* Summary of **AUE2_2013**

    * Reacting equally well to different types of drift
    
    * The proposed algorithm was also optimized for memory usage by restricting ensemble size and incorporating a simple inner-component pruning mechanism
    
    * Additional contributions of AUE2 include the proposal of a new component weighting function and a cost-effective candidate weight
    
--- 

### 2020年3月23日~31日

#### Reading:

* Leveraging Bagging for Evolving Data Streams (**leveraging_bagging**)

* Adaptive random forests for evolving data stream classification

* Fast and Light Boosting for Adaptive Mining of Data Streams

* A Selective Detector Ensemble for Concept Drift Detection

* Three-layer concept drifting detection in text data streams

* Paired Learners for Concept Drift (**Paired_Learning**)

#### Done:

* ***Paired_Learners***

* ***leveraging_bagging*** (rectified)

#### Nodes:

* Summary of **Paired_Learning**