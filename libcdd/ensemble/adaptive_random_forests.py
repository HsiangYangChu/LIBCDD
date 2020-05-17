from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.drift_detection import ADWIN
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest


class AdaptiveRandomForest(AdaptiveRandomForest):
    """
    come from skmultiflow
    """

    def __init__(self,
                 n_estimators=10,
                 max_features='auto',
                 disable_weighted_vote=False,
                 lambda_value=6,
                 performance_metric='acc',
                 drift_detection_method: BaseDriftDetector = ADWIN(0.001),
                 warning_detection_method: BaseDriftDetector = ADWIN(0.01),
                 max_byte_size=33554432,
                 memory_estimate_period=2000000,
                 grace_period=50,
                 split_criterion='info_gain',
                 split_confidence=0.01,
                 tie_threshold=0.05,
                 binary_split=False,
                 stop_mem_management=False,
                 remove_poor_atts=False,
                 no_preprune=False,
                 leaf_prediction='nba',
                 nb_threshold=0,
                 nominal_attributes=None,
                 random_state=None):
        """AdaptiveRandomForest class constructor."""
        super().__init__(n_estimators,
                         max_features,
                         disable_weighted_vote,
                         lambda_value,
                         performance_metric,
                         drift_detection_method,
                         warning_detection_method,
                         max_byte_size,
                         memory_estimate_period,
                         grace_period,
                         split_criterion,
                         split_confidence,
                         tie_threshold,
                         binary_split,
                         stop_mem_management,
                         remove_poor_atts,
                         no_preprune,
                         leaf_prediction,
                         nb_threshold,
                         nominal_attributes,
                         random_state)