from __future__ import division

from numpy import dot
from numpy.random import RandomState
from numpy.testing import assert_allclose

from limix.qtl import LMM


def test_qtl_lmm():
    random = RandomState(1)

    N = 100
    S = 1000

    # generate data
    snps = (random.rand(N, S) < 0.2).astype(float)
    pheno = random.randn(N, 1)
    W = random.randn(N, 10)
    kinship = dot(W, W.T) / float(10)

    # run single-variant associaiton testing with LMM
    lmm = LMM(snps, pheno, kinship)
    pv = lmm.getPv()
    beta = lmm.getBetaSNP()
    beta_ste = lmm.getBetaSNPste()

    assert_allclose(
        pv[:, :5].ravel(),
        [0.85712431, 0.46681538, 0.58717204, 0.55894821, 0.19178414],
        rtol=1e-5,
        atol=1e-5)

    assert_allclose(
        beta.ravel()[:5],
        [0.0436056, -0.16947568, -0.11999516, 0.13877098, 0.28097339],
        rtol=1e-5,
        atol=1e-5)

    assert_allclose(
        beta_ste.ravel()[:5],
        [0.24220492, 0.23290171, 0.22101052, 0.23745709, 0.21525261],
        rtol=1e-5,
        atol=1e-5)

# def test_qtl_lmm1():
#     import numpy as np
#
#     random = RandomState(0)
#
#     N = 500
#     S = 1000
#     snps = (random.rand(N, S) < 0.2).astype(float)
#     w = random.randn(S)
#     w[-500:] = 0
#     pheno = snps.dot(w)
#     K = snps.dot(snps.T)
#     K /= K.diagonal().mean()
#     lmm = LMM(snps.copy(), pheno.copy(), K.copy())
#     lmm.process()
#
#     pv = lmm.getPv()
#     beta = lmm.getBetaSNP()
#     beta_ste = lmm.getBetaSNPste()
#
#     lml0 = lmm.NLL_0[0][0]
#     # 1740.414870219681
#
#     # import limix_legacy.deprecated
#     # import limix_legacy.deprecated as dlimix_legacy
#
#     from glimix_core.lmm import LMM as HLMM
#     from numpy_sugar.linalg import economic_qs
#
#     QS = economic_qs(K)
#
#     lmm = HLMM(pheno.ravel(), np.ones((N, 1)), QS)
#     lmm.learn(progress=True)
#     lml_null = lmm.lml()
#     flmm = lmm.get_fast_scanner()
#     lml_alts, effect_sizes = flmm.fast_scan(snps)
#     # -1740.3165959073499
#
#     from numpy import asarray
#     _lrs = -2 * lml_null + 2 * asarray(lml_alts)
#     from scipy.stats import chi2
#     _chi2 = chi2(df=1)
#     pvalues = _chi2.sf(_lrs)
#
#     import pdb; pdb.set_trace()
#     # (Pdb) !pvalues[:3], pvalues[-3:]
#     # (array([ 0.01378375,  0.70275829,  0.736809  ]), array([ 0.80680504,  0.78269143,  0.57084244]))
#     # (Pdb) !pv.ravel()[:3], pv.ravel()[-3:]
#     # (array([ 0.01202286,  0.70815692,  0.72339446]), array([ 0.812167  ,  0.79096445,  0.57440581]))
