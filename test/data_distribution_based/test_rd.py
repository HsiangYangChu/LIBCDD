import numpy as np
import time

from libcdd.data_distribution_based import RD
from libcdd.draw import draw_concise_diagram


size = 80
concept_num = 10

l = 0
r = 1
dia = 100
data = []
for i in range(concept_num):
    data += np.random.uniform(l, r, size).tolist()
    l += dia
    r += dia

t1 = time.time()
rd = RD(window_size=20, n=20)
t2 = time.time()

print(t2 - t1)

indexes = []
rds = []

for i in range(len(data)):
    rd.add_element(data[i])
    rds.append(rd.rd)
    if rd.in_concept_change:
        indexes.append(i)

print(rds)

draw_concise_diagram(np.asarray(rds))

print(indexes)

