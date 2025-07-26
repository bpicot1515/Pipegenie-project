# pipegenie/metalearning/metafeatures.py

from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import skew, kurtosis
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class BaseMetafeatureCalculator(ABC):
    """
    Abstract base class for meta-feature calculation.
    
    Provides a method to calculate features common to all supervised learning tasks.
    """

    def calculate_common_features(self, X: "ArrayLike", y: "ArrayLike") -> Dict[str, float]:
        """Calculates features common to both classification and regression."""
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        
        features = {
            "NumberOfInstances": float(n_samples),
            "NumberOfFeatures": float(n_features),
            "DatasetRatio": float(n_features) / float(n_samples) if n_samples > 0 else 0.0,
        }

        # Feature statistics (works for both tasks)
        # To avoid high computation cost, we sample up to 100 features
        if n_features > 100:
            rng = np.random.default_rng()
            feature_indices = rng.choice(n_features, 100, replace=False)
            X_sample = X[:, feature_indices]
        else:
            X_sample = X

        with np.errstate(invalid='ignore'): # Ignore warnings for columns with no variation
            features["SkewnessMean"] = float(np.nanmean(skew(X_sample, axis=0)))
            features["KurtosisMean"] = float(np.nanmean(kurtosis(X_sample, axis=0)))

        return features

    @abstractmethod
    def calculate(self, X: "ArrayLike", y: "ArrayLike") -> Dict[str, float]:
        """
        Computes the full set of meta-features for the dataset (X, y).
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the 'calculate' method.")


class ClassificationMetafeatureCalculator(BaseMetafeatureCalculator):
    """Calculates meta-features specific to classification tasks."""

    def calculate(self, X: "ArrayLike", y: "ArrayLike") -> Dict[str, float]:
        """
        Computes meta-features for a classification dataset.

        This includes common features plus classification-specific ones like
        number of classes and class imbalance metrics.
        """
        # 1. Get common features from the base class
        features = self.calculate_common_features(X, y)
        
        # 2. Add classification-specific features
        y = np.asarray(y)
        n_samples = len(y)
        classes, counts = np.unique(y, return_counts=True)
        
        features["NumberOfClasses"] = float(len(classes))
        
        if len(counts) > 1:
            min_class_count = float(np.min(counts))
            max_class_count = float(np.max(counts))
            features["MinClassProbability"] = min_class_count / n_samples
            features["MaxClassProbability"] = max_class_count / n_samples
            features["ClassImbalanceRatio"] = min_class_count / max_class_count
        else:
            features["MinClassProbability"] = 1.0
            features["MaxClassProbability"] = 1.0
            features["ClassImbalanceRatio"] = 1.0
            
        return features


class RegressionMetafeatureCalculator(BaseMetafeatureCalculator):
    """Calculates meta-features specific to regression tasks."""

    def calculate(self, X: "ArrayLike", y: "ArrayLike") -> Dict[str, float]:
        """
        Computes meta-features for a regression dataset.

        This includes common features plus regression-specific ones describing
        the target variable's distribution.
        """
        # 1. Get common features from the base class
        features = self.calculate_common_features(X, y)

        # 2. Add regression-specific features (properties of the target)
        y = np.asarray(y)
        with np.errstate(invalid='ignore'):
            features["TargetSkewness"] = float(skew(y))
            features["TargetKurtosis"] = float(kurtosis(y))
            features["TargetMean"] = float(np.nanmean(y))
            features["TargetStdev"] = float(np.nanstd(y))

        return features