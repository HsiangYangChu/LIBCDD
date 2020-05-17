import os
import numpy as np
from libcdd.data_distribution_based.ita import ITA


def test_ita(test_path):
    """
    ITA drift detection test.
    The first half of the stream contains a sequence corresponding to a normal distribution of integers from 0 to 1.
    From index 999 to 1999 the sequence is a normal distribution of integers from 0 to 7.

    """
    ita = ITA()
    test_file = os.path.join(test_path, 'drift_stream.npy')
    data_stream = np.load(test_file)
    expected_indices = [1023, 1055, 1087, 1151]
    detected_indices = []

    for i in range(data_stream.size):
        # print(data_stream[i])
        ita.add_element(data_stream[i])
        if ita.detected_change():
            detected_indices.append(i)

    print(detected_indices)

    assert detected_indices == expected_indices

    expected_info = "ita(delta=0.002)"
    assert ita.get_info() == expected_info

test_ita('/Users/yang/Documents/ConceptDriftDetectionLib/test/data_distribution_based/')