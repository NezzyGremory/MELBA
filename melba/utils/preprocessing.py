"""Training-fitted standardization and quadratic business feature mapping."""

from __future__ import annotations

import numpy as np


class BusinessFeatureMap:
    """Standardize inputs and optionally add squared and pairwise terms."""

    def __init__(self, quadratic_terms: bool = True) -> None:
        self.quadratic_terms = quadratic_terms
        self.n_features_in_: int | None = None
        self.input_mean_: np.ndarray | None = None
        self.input_scale_: np.ndarray | None = None
        self.term_mean_: np.ndarray | None = None
        self.term_scale_: np.ndarray | None = None

    def _expand(self, standardized: np.ndarray) -> np.ndarray:
        if not self.quadratic_terms:
            return standardized
        count = standardized.shape[1]
        quadratic = [
            standardized[:, left] * standardized[:, right]
            for left in range(count)
            for right in range(left, count)
        ]
        return np.column_stack((standardized, *quadratic))

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        self.n_features_in_ = features.shape[1]
        with np.errstate(over="ignore", invalid="ignore"):
            self.input_mean_ = np.mean(features, axis=0)
            input_scale = np.std(features, axis=0)
        if not np.isfinite(self.input_mean_).all() or not np.isfinite(input_scale).all():
            raise ValueError("X values are too large to scale safely.")
        self.input_scale_ = np.where(input_scale > np.finfo(float).eps, input_scale, 1.0)

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            standardized = (features - self.input_mean_) / self.input_scale_
            expanded = self._expand(standardized)
            self.term_mean_ = np.mean(expanded, axis=0)
            term_scale = np.std(expanded, axis=0)
        if (
            not np.isfinite(expanded).all()
            or not np.isfinite(self.term_mean_).all()
            or not np.isfinite(term_scale).all()
        ):
            raise ValueError("X values are too large to scale safely.")
        self.term_scale_ = np.where(term_scale > np.finfo(float).eps, term_scale, 1.0)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mapped = (expanded - self.term_mean_) / self.term_scale_
        if not np.isfinite(mapped).all():
            raise ValueError("X values are too large to scale safely.")
        return mapped

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.n_features_in_ is None or self.input_mean_ is None:
            raise RuntimeError("The feature map must be fitted before transform.")
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features than the data used for training; "
                f"expected {self.n_features_in_}, got {features.shape[1]}."
            )
        assert self.input_scale_ is not None
        assert self.term_mean_ is not None
        assert self.term_scale_ is not None
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            standardized = (features - self.input_mean_) / self.input_scale_
            expanded = self._expand(standardized)
            mapped = (expanded - self.term_mean_) / self.term_scale_
        if not np.isfinite(mapped).all():
            raise ValueError("X values are too large to transform safely.")
        return mapped

