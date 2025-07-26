# pipegenie/preprocessing/_balancing.py

#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#
# This component is inspired by the Balancing component in auto-sklearn.
#

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class Balancing:
    """
    A marker component for balancing imbalanced class distributions.

    This component does not transform the data itself. Instead, it serves as a
    marker for the evolutionary engine to apply a weighting strategy during
    the evaluation of the pipeline.

    Parameters
    ----------
    strategy : str, {'none', 'weighting'}, default='none'
        The balancing strategy. If 'weighting', sample weights will be
        calculated and passed to the final estimator's `fit` method.
    """

    def __init__(self, strategy: str = "none"):
        self.strategy = strategy

    def fit(self, X: 'ArrayLike', y: 'Optional[ArrayLike]' = None) -> 'Balancing':
        """
        This method does nothing and is for scikit-learn compatibility.
        """
        return self

    def transform(self, X: 'ArrayLike') -> 'ArrayLike':
        """
        This method returns the data unchanged.
        """
        return X

    def __str__(self) -> str:
        return f"Balancing(strategy='{self.strategy}')"