from skmultiflow.drift_detection.page_hinkley import PageHinkley


class PageHinkley(PageHinkley):

    """
    come from skmultiflow

    """

    def __init__(self, min_instances=30, delta=0.005, threshold=50, alpha=1 - 0.0001):
        super().__init__(min_instances, delta, threshold, alpha)

