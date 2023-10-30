from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


class CensoredShuffleSplit(StratifiedShuffleSplit):
    """

    """
    def _iter_indices(self, X, y, groups=None):
        return super()._iter_indices(X, y["event"], groups)


class CensoredKFold(StratifiedKFold):
    """

    """
    def _iter_test_masks(self, X, y=None, groups=None):
        return super()._iter_test_masks(X, y["event"], groups)
