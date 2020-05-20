import numpy as np
from libcdd.error_rate_based import *
#新建一个检测器
detector = HDDM_A()
#获取2000个数据，其中前后1000个数据分别服从不同的正态分布
#以0为决策边界
np.random.seed(1)
mu, std = 0, 0.1      # 均值和标准差
data1 = np.random.normal(mu, std, 1000) > 0
data1 = data1.astype(int)
mu, std = 0.5, 0.1
data2 = np.random.normal(mu, std, 1000) > 0
data2 = data2.astype(int)
data_stream = np.concatenate((data1, data2))
#检测过程
detected_indices = []
for i in range(data_stream.size):
    detector.add_element(data_stream[i])
    if detector.detected_change():
        print(str(i)+"时刻发生了概念漂移")
#程序的运行结果为：1049时刻发生了概念漂移