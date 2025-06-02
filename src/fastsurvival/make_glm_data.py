import os
import numpy as np
import pandas as pd

'''
Code adapted from ABESS: https://github.com/abess-team/abess/blob/master/python/abess/datasets.py
'''


def sample(p, k):
    full = np.arange(p)
    select = sorted(np.random.choice(full, k, replace=False))
    return select

class make_glm_data:
    """
    Generate a dataset with single response.

    Parameters
    ----------
    n: int
        The number of observations.
    p: int
        The number of predictors of interest.
    k: int
        The number of nonzero coefficients in the
        underlying regression model.
    family: {gaussian, binomial, poisson, gamma, cox}
        The distribution of the simulated response.
        "gaussian" for univariate quantitative response,
        "binomial" for binary classification response,
        "poisson" for counting response,
        "gamma" for positive continuous response,
        "cox" for left-censored response.
    rho: float, optional, default=0
        A parameter used to characterize the pairwise
        correlation in predictors.
    corr_type: string, optional, default="const"
        The structure of correlation matrix.
        "const" for constant pairwise correlation,
        "exp" for pairwise correlation with exponential decay.
    sigma: float, optional, default=1
        The variance of the gaussian noise.
        It would be unused if snr is not None.
    coef_: array_like, optional, default=None
        The coefficient values in the underlying regression model.
    censoring: bool, optional, default=True
        For Cox data, it indicates whether censoring is existed.
    c: int, optional, default=1
        For Cox data and censoring=True, it indicates the maximum
        censoring time.
        So that all observations have chances to be censored at (0, c).
    scal: float, optional, default=10
        The scale of survival time in Cox data.
    snr: float, optional, default=None
        A numerical value controlling the signal-to-noise ratio (SNR)
        in gaussian data.
    class_num: int, optional, default=3
        The number of possible classes in oridinal dataset, i.e.
        :math:`y \in \{0, 1, 2, ..., \text{class_num}-1\}`

    Attributes
    ----------
    x: array-like, shape(n, p)
        Design matrix of predictors.
    y: array-like, shape(n,)
        Response variable.
    coef_: array-like, shape(p,)
        The coefficients used in the underlying regression model.
        It has k nonzero values.

    Notes
    -----
    The output, whose type is named ``data``, contains three elements:
    ``x``, ``y`` and ``coef_``, which correspond the variables, responses
    and coefficients, respectively.

    Each row of ``x`` or ``y`` indicates a sample and is independent to the
    other.

    We denote :math:`x, y, \beta` for one sample in the math formulas below.

    * Cox PH Survival Analysis

        * Usage: ``family='cox'[, scal=..., censoring=..., c=...]``
        * Model: :math:`y=\min(t,C)`,
          where :math:`t = \left[-\dfrac{\log U}{\exp(X \beta)}\right]^s,\
          U\sim N(0,1),\ s=\dfrac{1}{\text{scal}}` and
          censoring time :math:`C\sim U(0, c)`.

            * the coefficient :math:`\beta\sim U[2m, 10m]`,
              where :math:`m = 5\sqrt{2\log p/n}`;
            * the scale of survival time :math:`\text{scal} = 10`;
            * censoring is enabled, and max censoring time :math:`c=1`.

    """

    def __init__(self, n, p, k, family, rho=0, corr_type="const", sigma=1,
                 coef_=None,
                 censoring=True, c=1, scal=10, snr=None, class_num=3):
        self.n = n
        self.p = p
        self.k = k
        self.family = family

        if corr_type == "exp":
            # generate correlation matrix with exponential decay
            R = np.zeros((p, p))
            for i in range(p):
                for j in range(i, p):
                    R[i, j] = rho ** abs(i - j)
            R = R + R.T - np.identity(p)
        elif corr_type == "const":
            # generate correlation matrix with constant correlation
            R = np.ones((p, p)) * rho
            for i in range(p):
                R[i, i] = 1
        else:
            raise ValueError(
                "corr_type should be \'const\' or \'exp\'")

        x = np.random.multivariate_normal(mean=np.zeros(p), cov=R, size=(n,))

        nonzero = sample(p, k)
        Tbeta = np.zeros(p)
        sign = np.random.choice([1, -1], k)

        if family == "cox":
            m = 5 * sigma * np.sqrt(2 * np.log(p) / n)
            if coef_ is None:
                Tbeta[nonzero] = np.random.uniform(2 * m, 10 * m, k) * sign
            else:
                Tbeta = coef_

            time = np.power(-np.log(np.random.uniform(0, 1, n)) /
                            np.exp(np.matmul(x, Tbeta)), 1 / scal)

            if censoring:
                ctime = c * np.random.uniform(0, 1, n)
                status = (time < ctime) * 1
                censoringrate = 1 - sum(status) / n
                print("censoring rate:" + str(censoringrate))
                for i in range(n):
                    time[i] = min(time[i], ctime[i])
            else:
                status = np.ones(n)
                print("no censoring")

            y = np.hstack((time.reshape((-1, 1)), status.reshape((-1, 1))))
        
        else:
            raise ValueError(
                "Family should be \'cox\'.")
        self.x = x
        self.y = y
        self.coef_ = Tbeta

if __name__ == "__main__":

        n = 1200
        p = 1200
        k = 15
        family = 'cox'
        rho = 0.9
        corr_type = 'exp'
        coef_ = np.zeros(p)
        coef_[0:p:(p // k)] = 1
        true_coef = coef_

        print(coef_.nonzero())

        data = make_glm_data(n=n, p=p, k=k, family=family, rho=rho, coef_=coef_, corr_type=corr_type, censoring=True, c=1, scal=10)

        time_series = data.y[:, 0]
        event_series = data.y[:, 1]
        print("time_series", time_series)
        print("event_series", event_series)
        print("X", data.x)