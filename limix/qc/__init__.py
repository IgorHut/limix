r"""
***************
Quality control
***************

Transformation
^^^^^^^^^^^^^^

.. autofunction:: limix.qc.boxcox
.. autofunction:: limix.qc.mean_standardize
.. autofunction:: limix.qc.quantile_gaussianize

"""

from .transformation import (boxcox, mean_standardize, quantile_gaussianize,
                             regress_out, remove_dependent_cols)

__all__ = [
    'boxcox', 'mean_standardize', 'quantile_gaussianize', 'regress_out',
    'remove_dependent_cols'
]