# Vendored from https://github.com/vincent-leguen/DILATE (MIT licence)
# Le Guen & Thome, "Shape and Time Distortion Loss for Training Deep Time Series
# Forecasting Models", NeurIPS 2019.
from .dilate_loss import dilate_loss

__all__ = ["dilate_loss"]
