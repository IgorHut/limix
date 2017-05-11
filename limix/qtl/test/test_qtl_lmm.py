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
