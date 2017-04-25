from __future__ import division

from numpy import asarray, ones

from numpy_sugar.linalg import economic_qs
from glimix_core.glmm import GLMM

def qtl_test_glmm(
        snps,
        pheno,
        lik,
        K,
        covs=None,
        test='lrt',
        NumIntervalsDeltaAlt=100,
        searchDelta=False,
        verbose=None):
    """
    Wrapper function for univariate single-variant association testing
    using a generalised linear mixed model.

    Args:
        snps (array_like):
            `N` individuals by `S` SNPs.
        pheno (tuple, array_like):
            Either a tuple of two arrays of `N` individuals each (Binomial
            phenotypes) or an array of `N` individuals (Poisson or Bernoulli
            phenotypes). It does not support missing values yet.
        lik ({'bernoulli', 'binomial', 'poisson'}):
            Sample likelihood describing the residual distribution.
        K (array_like):
            `N` by `N` covariance matrix (e.g., kinship coefficients).
        covs (array_like, optional):
            `N` individuals by `D` covariates.
            By default, ``covs`` is a (`N`, `1`) array of ones.
        test ({'lrt'}, optional):
            Likelihood ratio test (default).
        NumIntervalsDeltaAlt (int, optional):
            number of steps for delta optimization on the alternative model.
            Requires ``searchDelta=True`` to have an effect.
        searchDelta (bool, optional):
            if ``True``, delta optimization on the alternative model is
            carried out. By default ``searchDelta`` is ``False``.
        verbose (bool, optional):
            if ``True``, details such as runtime are displayed.

    Returns:
        :class:`limix.qtl.LMM`: LIMIX LMM object
    """
    snps = asarray(snps, float)

    if covs is None:
        covs = ones((snps.shape[0], 1))
    else:
        covs = asarray(covs, float)

    K = asarray(K, float)

    if isinstance(pheno, (tuple, list)):
        y = tuple([asarray(p, float) for p in pheno])
    else:
        y = asarray(pheno, float)

    QS = economic_qs(K)
    glmm = GLMM(y, lik, covs, QS)
    glmm.feed().maximize(progress=verbose)

    # lmm_ = LMM(
    #     snps=snps,
    #     pheno=pheno,
    #     K=K,
    #     covs=covs,
    #     test=test,
    #     NumIntervalsDelta0=NumIntervalsDelta0,
    #     NumIntervalsDeltaAlt=NumIntervalsDeltaAlt,
    #     searchDelta=searchDelta,
    #     verbose=verbose)
    # return lmm_
    return None
