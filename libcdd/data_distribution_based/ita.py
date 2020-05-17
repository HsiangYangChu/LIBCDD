import numpy as np
import math
import time

from .kdqtree import KDQTree
from .base_distribution_detector import BaseDistributionDetector


class ITA(BaseDistributionDetector):

    def __init__(self, window_size=200, side=pow(2, -10), leaf_size=100, persistence_factor=0.05, asl=0.01,
                 bootstrap_num=500):
        super().__init__()
        self.window_size = window_size
        self.side = side
        self.leaf_size = leaf_size
        self.persistence_factor = persistence_factor
        self.asl = asl
        self.bootstrap_num = bootstrap_num
        self.threshold = window_size * persistence_factor

        self.window_his = None
        self.index = None
        self.kdqtree = None
        self.leafs = None
        self.values = None
        self.kl_distance = None
        self.higher = None
        self.count = None
        self.number_sum = None
        self.reset()

    def reset(self):
        super().reset()
        self.window_his = []
        self.window_queue = [0 for _ in range(self.window_size)]
        self.index = 0
        self.number_sum = 0
        self.count = 0
        self.kdqtree = None
        self.leafs = None
        self.kl_distance = None

    def add_element(self, input_value):
        if self.in_concept_change:
            self.reset()

        if input_value.ndim == 0:
            input_value = np.asarray([input_value])

        if len(self.window_his) < self.window_size:
            self.window_his.append(input_value)
            if len(self.window_his) == self.window_size:
                self.get_new_kdqtree()
                time1 = time.time()
                self.higher = self.bootstrap(data=np.asarray(self.window_his), B=self.bootstrap_num, c=self.asl,
                                             func=self.get_kl_distance)
                time2 = time.time()
                print("Bootstrap time: {}", time2 - time1)
        else:
            self.number_sum += 1
            if self.number_sum > self.window_size:
                changed_leaf1 = self.window_queue[self.index]
                self.leafs[changed_leaf1] -= 1
                self.window_queue[self.index] = self.kdqtree.query(np.asarray([input_value]))[0][0]
                changed_leaf2 = self.window_queue[self.index]
                self.leafs[changed_leaf2] += 1

                self.kl_distance -= self.values[changed_leaf1]
                pv = self.kdqtree.nodes_per_leaf[changed_leaf1]
                qv = self.leafs[changed_leaf1]
                self.values[changed_leaf1] = (pv + 0.5) * math.log((pv + 0.5) / (qv + 0.5))
                self.kl_distance += self.values[changed_leaf1]

                self.kl_distance -= self.values[changed_leaf2]
                pv = self.kdqtree.nodes_per_leaf[changed_leaf2]
                qv = self.leafs[changed_leaf2]
                self.values[changed_leaf2] = (pv + 0.5) * math.log((pv + 0.5) / (qv + 0.5))
                self.kl_distance += self.values[changed_leaf2]

                self.index = (self.index + 1) % self.window_size
            else:
                self.window_queue[self.index] = self.kdqtree.query(np.asarray([input_value]))[0][0]
                self.leafs[self.window_queue[self.index]] += 1
                self.index = (self.index + 1) % self.window_size
                if self.number_sum == self.window_size:
                    self.kl_distance = 0
                    for i in range(len(self.leafs)):
                        pv = self.kdqtree.nodes_per_leaf[i]
                        qv = self.leafs[i]
                        self.values[i] = (pv + 0.5) * math.log((pv + 0.5) / (qv + 0.5))
                        self.kl_distance += self.values[i]

            if self.kl_distance is None:
                return
            # print(self.kl_distance)
            if self.kl_distance > self.higher:
                self.count += 1
                if self.count > self.threshold:
                    print("changed")
                    self.in_concept_change = True
            else:
                self.count = 0

    def get_new_kdqtree(self):
        self.kdqtree = KDQTree(X=np.asarray(self.window_his), leaf_size=self.leaf_size, min_side=self.side)
        self.leafs = [0 for _ in range(len(self.kdqtree.nodes_per_leaf))]
        self.values = [0 for _ in range(len(self.kdqtree.nodes_per_leaf))]

    def get_kl_distance(self, X1, X2):
        kdqtree = KDQTree(X1, leaf_size=self.leaf_size, min_side=self.side)
        leafs = [0 for _ in range(len(kdqtree.nodes_per_leaf))]
        leaf_id_all = kdqtree.query(X2)
        for id in leaf_id_all:
            leafs[id[0]] += 1
        kl_distance = 0.0
        for i in range(len(leafs)):
            pv = kdqtree.nodes_per_leaf[i]
            qv = leafs[i]
            kl_distance = (pv + 0.5) * math.log((pv + 0.5) / (qv + 0.5))
        return kl_distance

    def bootstrap(self, data, B, c, func):
        # B = 1
        array = np.array(data)
        n = len(array)
        sample_result_arr = []
        for i in range(B):
            index_arr = np.random.randint(0, n, size=n)
            data_sample1 = array[index_arr]
            index_arr = np.random.randint(0, n, size=n)
            data_sample2 = array[index_arr]
            sample_result = func(data_sample1, data_sample2)
            sample_result_arr.append(sample_result)

        k2 = int(B * (1 - c))
        auc_sample_arr_sorted = sorted(sample_result_arr)
        higher = auc_sample_arr_sorted[k2]
        return higher
