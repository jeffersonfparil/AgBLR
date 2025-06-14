import numpy as np
import numpy.typing as npt
from scipy.stats import norm, uniform, laplace, beta
from scipy.stats import multivariate_normal
from typing import Self


class CovarianceMatrix:
    def __init__(self, p: int, label_prefix: str = "param") -> Self:
        self.V = np.eye(p, p)
        d = len(str(p))
        self.labels = [f"{label_prefix}-{str(i + 1).zfill(d)}" for i in range(p)]

    def simulate_spherical(self, s: float = 1.00) -> Self:
        n = self.V.shape[0]
        self.V = np.eye(n, n) * s
        return self

    def simulate_simple(self, x: npt.NDArray[1]) -> Self:
        if len(x) != self.V.shape[0]:
            raise ValueError(
                "Input vector length must match the covariance matrix size."
            )
        self.V = np.outer(x, x) / len(x)
        return self

    def simulate_diagonal(self, x: npt.NDArray[1], seed: int = 42) -> Self:
        if len(x) != self.V.shape[0]:
            raise ValueError(
                "Input vector length must match the covariance matrix size."
            )
        np.random.seed(seed)
        n = len(x)
        diagonal_elements = np.random.uniform(min(x), max(x), size=n)
        self.V = np.diag(diagonal_elements)
        return self

    def simulate_random(self, x: npt.NDArray[1], seed: int = 42) -> Self:
        if len(x) != self.V.shape[0]:
            raise ValueError(
                "Input vector length must match the covariance matrix size."
            )
        np.random.seed(seed)
        n = len(x)
        R = np.random.rand(n, n)
        R = min(x) + (max(x) - min(x)) * R
        self.V = (R @ R.T) / n
        return self

    def simulate_autocorrelation(self, x: npt.NDArray[1]) -> Self:
        if len(x) != self.V.shape[0]:
            raise ValueError(
                "Input vector length must match the covariance matrix size."
            )
        n = len(x)
        for i in range(n):
            for j in range(i, n):
                lag = abs(i - j)
                # a = np.correlate(x[i:], np.roll(x[j:], lag), mode="valid")[0] / n
                x1 = x[0:(n-lag)]
                x2 = np.roll(x, lag)[0:(n-lag)]
                a = np.correlate(x1, x2, mode="valid")[0] / n
                self.V[i, j] = a
                self.V[j, i] = a
        return self

    def simulate_kinship(
        self,
        x: npt.NDArray[1],
        k: int = 3,        # Number of kinship groups
        r: float = 0.25,   # Correlation strength within groups
        eps: float = 0.01, # Small random noise to add
        seed: int = 42,
    ) -> Self:
        if len(x) != self.V.shape[0]:
            raise ValueError(
                "Input vector length must match the covariance matrix size."
            )
        n = len(x)
        if n < k:
            raise ValueError("Input vector must have at least k elements.")
        np.random.seed(seed)
        groupings = np.repeat(np.arange(k), np.ceil(n / k))[:n]
        np.random.shuffle(groupings)
        for i in range(k):
            (idx,) = np.where(groupings == i)
            row_indices, col_indices = np.meshgrid(idx, idx)
            self.V[row_indices, col_indices] += r
        self.V += np.random.normal(0, eps, (n, n))
        return self

    def inflate_diagonal(
        self, s: float = 0.01, tol: float = 1e-7, max_iter: int = 10
    ) -> Self:
        d = np.abs(np.linalg.det(self.V))
        for _ in range(max_iter):
            if d < tol:
                self.V += np.eye(self.V.shape[0]) * (1 + s)
                d = np.linalg.det(self.V)
            else:
                break
        if d < tol:
            raise ValueError(
                "Matrix is not positive definite after inflation for a maximum of {} iterations. Please consider increase `s` and/or `max_iter`".format(
                    max_iter
                )
            )
        return self


class Effects:
    def __init__(
        self,
        p: int,
        dmeans=[norm, uniform, laplace, beta][0],
        pmeans: tuple[float, float] = (2.0, 1.0),
        label_prefix: str = "param",
        seed: int = 42,
    ):
        x = dmeans(pmeans[0], pmeans[1]).rvs(size=p)
        K = CovarianceMatrix(p=p, label_prefix=label_prefix)
        K.simulate_simple(x)
        if np.abs(np.linalg.det(K.V)) <= 1e-7:
            K.inflate_diagonal(s=0.01, tol=1e-7, max_iter=10)
        MVN = multivariate_normal(mean=x, cov=K.V, allow_singular=False, seed=seed)
        self.labels = K.labels
        self.V = K.V
        self.b = MVN.rvs(size=1)
        return self
