import numpy as np
from libcdd.error_rate_based.dde import DDE


dm = DDE()

data_stream = np.random.randint(2, size=2000)

for i in range(999, 1500):
    data_stream[i] = 0


for i in range(2000):
    dm.add_element(data_stream[i])
    # if dm.detected_warning_zone():
    #     print('Warning zone has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    if dm.detected_change():
        print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))