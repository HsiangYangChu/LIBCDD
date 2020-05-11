from abc import ABCMeta, abstractmethod
from skmultiflow.core import BaseSKMObject

class BaseDistributionDetector(BaseSKMObject, metaclass=ABCMeta):
    """ Abstract Distribution Detector

    Any Distribution detector class should follow this minimum structure in
    order to allow interchangeability between all change detection
    methods.

    Raises
    ------
    NotImplementedError. All child classes should implement the
    get_info function.

    """

    estimator_type = "distribution_detector"

    def __init__(self):
        super().__init__()
        self.in_concept_change = None
        self.statistic = None
        

    def reset(self):
        """ reset

        Resets the change detector parameters.

        """
        self.in_concept_change = False
        self.statistic = -1

    def detected_change(self):
        """ detected_change

        This function returns whether concept drift was detected or not.

        Returns
        -------
        bool
            Whether concept drift was detected or not.

        """
        return self.in_concept_change

    def get_statistic(self):
        """ get_length_estimation

        Returns the length estimation.

        Returns
        -------
        int
            The length estimation

        """
        return self.statistic

    @abstractmethod
    def add_element(self, input_value):
        """ add_element

        Adds the relevant data from a sample into the change detector.

        Parameters
        ----------
        input_value: Not defined
            Whatever input value the change detector takes.

        Returns
        -------
        BaseDriftDetector
            self, optional

        """
        raise NotImplementedError