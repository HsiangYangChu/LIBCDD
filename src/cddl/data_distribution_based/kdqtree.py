import copy as cp

from skmultiflow.core import BaseSKMObject
from skmultiflow.lazy.distances import mixed_distance, euclidean_distance
from skmultiflow.utils.utils import *

class KDQTree(BaseSKMObject):

    def __init__(self, X, categorical_list=None, leaf_size=100, min_side=pow(2, -10)):
        super().__init__()

        self.X = np.asarray(X)
        if self.X.ndim != 2:  # 数组维度
            raise ValueError("X should be a matrix, or array-like, of shape (n_samples, n_features).")

        self.X = self.X.astype(np.float64)

        self.n_samples, self.n_features = self.X.shape

        self.categorical_list = categorical_list
        self.leaf_size = leaf_size
        self.min_side = min_side

        self.nodes_per_leaf = []
        self.root = None
        self.maxes = np.amax(self.X, axis=0)
        self.mins = np.amin(self.X, axis=0)
        self.cur = [1.0 for _ in range(self.n_features)]
        self.aux_query_X = None
        indexes = [i for i in range(self.n_samples)]
        self.root = self.__build(0, indexes)

    def __build(self, col, indexes):

        root = KDQTreeNode()
        root.split_axis = col
        maxval = self.maxes[col]
        minval = self.mins[col]
        size = len(indexes)

        if self.cur[col] < self.min_side or size < self.leaf_size:
            root.is_leaf = True
            root.id = len(self.nodes_per_leaf)
            self.nodes_per_leaf.append(size)
            return root

        midval = (maxval + minval) / 2
        root.split_value = midval
        root.split_axis = col

        left_indexes = []
        right_indexes = []
        for row in indexes:
            if self.X[row, col] > midval:
                right_indexes.append(row)
            else:
                left_indexes.append(row)

        # Create left son
        self.cur[col] /= 2
        self.maxes[col] = midval
        root.left_son = self.__build((col+1)%self.n_features, left_indexes)
        self.maxes[col] = maxval

        # Create right son
        self.mins[col] = midval
        root.right_son = self.__build((col+1)%self.n_features, right_indexes)
        self.mins[col] = minval
        self.cur[col] *= 2
        return root

    def query(self, X):

        r, c = get_dimensions(X)
        leaf_id_all = []
        for i in range(r):
            leaf_ids = []
            self.aux_query_X = X[i].astype(np.float64)
            leaf_ids.append(self._query(self.root))
            leaf_id_all.append(leaf_ids)
        return leaf_id_all

    def _query(self, root):

        if root.is_leaf:
            return root.id

        if self.aux_query_X[root.split_axis]<root.split_value:
            return self._query(root.left_son)
        else:
            return self._query(root.right_son)

class KDQTreeNode(object):

    def __init__(self):

        self.left_son = None
        self.right_son = None
        self.split_axis = None
        self.split_value = None
        self.is_leaf = False
        self.id = None




