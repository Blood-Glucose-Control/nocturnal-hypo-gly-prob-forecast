# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: [Add your contact information]

"""Discrete wavelet transform."""

import numpy as np
import pandas as pd
import pywt

from sktime.transformations.base import BaseTransformer

__author__ = ["Phiruby"]


class WaveletTransformer(BaseTransformer):
    """
    Applies the wavelet transform

    Uses pywt's implementations to perform DWT (discrete wavelet transform)

    Parameters
    ----------
    window_len: the window length to apply wavelet transform per
        Eg: if this is 14, applies wavelet transform on a series partitioned into blocks of 14 rows
    wavelet: string (default is "sym16")
        The name of the wavelet to use. See pywt's documentation for list of all available wavelets
    threshold: int / float (default is computed based on data size)
        The threshold value used to eliminate detail coefficients in the wavelet transform
    num_levels: int (default=5)
        The number of times to repeatedly apply the wavelet transform

    Examples
    --------
    >>> from sktime.transformations.panel.dwt import DWTTransformer
    >>> from sktime.datasets import load_airline
    >>> from sktime.datatypes import convert
    >>>
    >>> y = load_airline()
    >>> y = convert(y, to="Panel")
    >>> transformer = WaveletTransformer(num_levels=3)
    >>> y_transformed = transformer.fit_transform(y)
    """

    _tags = {
        "authors": "Phiruby",
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "pd.Series",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "pd.Series",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": True,
        "remember_data": False,
    }

    def __init__(self, window_len, wavelet="sym16", threshold=None, num_levels=3):
        self.num_levels = num_levels
        self.window_len = window_len
        self.wavelet = wavelet
        self.threshold = threshold
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of X must contain pandas.Series
            Data to fit transform to
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        X: a new pd.Series with wavelet transform applied on each window
        """
        self._check_parameters()

        # Get the wavelet coefficients, remove detailed coeffs (thresholding), reconstruct wave,
        # and return!
        Xt = X.copy()
        Xt = self.apply_wavelet(X)
        return Xt

    def apply_wavelet(self, series):
        # Partition the series into non-overlapping windows
        windows = self.partition_series(series)
        transformed_series = []

        for window in windows:
            # NOTE: I am unsure why we have to slice [:len(window)]... DWT should return same size
            # but usually is off by 1 here. This may affect outputs?
            transformed_series.append(
                self._extract_wavelet_coefficients(window)[: len(window)]
            )

        # Concatenate the results for each window
        return pd.Series(np.concatenate(transformed_series), index=series.index)

    def partition_series(self, series):
        """
        Partition the series into non-overlapping windows of size `window_len`.
        If the last window goes out of range, clip it to the max available range.

        Args:
            series: The time series data to partition
        """
        windows = []
        for i in range(0, len(series), self.window_len):
            # Clip the last window if it exceeds the range of the series
            window = series[i : min(i + self.window_len, len(series))]
            windows.append(window)

        return windows

    def _extract_wavelet_coefficients(self, data):
        """
        Extract wavelet coefficients from the data, and return the reconstructed data
        after deleting thresholded values

        Args:
            data: the input data
        """

        coeffs = pywt.wavedec(data, self.wavelet, level=5)

        threshold = self._get_coeff_threshold(data, coeffs)
        # apply thresholding to detail coefficients (keeping the approximation coefficients intact)- cA captures low-frequency stuff and cD's capture high frequency. We can smooth
        # the data by removing part of the cD's (i.e zero it)
        coeffs_new = coeffs.copy()
        coeffs_new[1:] = [pywt.threshold(c, threshold, mode="hard") for c in coeffs[1:]]

        # reconstruct the signal and plot!
        smoothed_signal = pywt.waverec(coeffs_new, self.wavelet)
        series = pd.Series(smoothed_signal)
        return series

    def _get_coeff_threshold(self, data, coeffs):
        """
        Gets the thresholding value based on the given data and coefficients. If hardcoded, returns that instead

        Args:
            data: the input data we are transforming
            coeffs: the coeffs obtained by calling pywt.wavedec
        """
        if self.threshold is None:
            sigma = np.median(np.max(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(data)))
        else:
            threshold = self.threshold
        return threshold

    def _check_parameters(self):
        """Check the values of parameters passed to DWT.

        Raises
        ------
        ValueError or TypeError if a parameter input is invalid.
        """
        if not isinstance(self.wavelet, str):
            raise TypeError(
                f"wavelet must be a 'str'. Found '{type(self.wavelet).__name__}' instead."
            )
        # None corresponds to dynamically computing the threshold
        if not isinstance(self.threshold, (int, float)) and self.threshold is not None:
            raise TypeError(
                f"threshold must be an 'int' or 'float'. Found '{type(self.threshold).__name__}' instead."
            )

        if self.threshold is not None and self.threshold < 0:
            raise ValueError("threshold must be non-negative.")

        if not isinstance(self.num_levels, int):
            raise TypeError(
                f"num_levels must be an 'int'. Found '{type(self.num_levels).__name__}' instead."
            )

        if self.num_levels < 0 or self.window_len < 0:
            raise ValueError("num_levels and window_len must be at least 0.")
