import numpy as np
import numpy.typing as npt
from scipy.stats import norm, uniform, laplace, beta
from scipy.stats import multivariate_normal
import functools
from typing import Self, Union, Callable


class CovarianceMatrix:
    def __init__(self, p: int, label_prefix: str = "param") -> Self:
        self.V = np.eye(p, p)
        d = len(str(p))
        self.labels = [f"{label_prefix}-{str(i + 1).zfill(d)}" for i in range(p)]

    def __repr__(self):
        print("Covariance Matrix:")
        with np.printoptions(precision=2, suppress=True):
            print(self.V)
        if len(self.labels) > 10:
            print(
                "Labels:", self.labels[0:5], "...", self.labels[-5 : len(self.labels)]
            )
        else:
            print("Labels:", self.labels)
        print("Determinant:", np.linalg.det(self.V))
        return "CovarianceMatrix(p={})".format(self.V.shape[0])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CovarianceMatrix):
            return NotImplemented
        return np.array_equal(self.V, other.V) and self.labels == other.labels

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, CovarianceMatrix):
            return NotImplemented
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((tuple(self.V.flatten()), tuple(self.labels)))

    def simulate_spherical(self, s: float = 1.00) -> Self:
        n = self.V.shape[0]
        self.V = np.eye(n, n) * s
        return self

    def simulate_simple(self, x: Union[None, npt.NDArray[1]] = None) -> Self:
        if x is None:
            x = np.diagonal(self.V)
        elif len(x) != self.V.shape[0]:
            raise ValueError(
                "Input vector length must match the covariance matrix size."
            )
        self.V = np.outer(x, x) / len(x)
        return self

    def simulate_diagonal(
        self, x: Union[None, npt.NDArray[1]] = None, seed: int = 42
    ) -> Self:
        if x is None:
            x = np.diagonal(self.V)
        elif len(x) != self.V.shape[0]:
            raise ValueError(
                "Input vector length must match the covariance matrix size."
            )
        np.random.seed(seed)
        n = len(x)
        diagonal_elements = np.random.uniform(min(x), max(x), size=n)
        self.V = np.diag(diagonal_elements)
        return self

    def simulate_random(
        self, x: Union[None, npt.NDArray[1]] = None, seed: int = 42
    ) -> Self:
        if x is None:
            x = np.diagonal(self.V)
        elif len(x) != self.V.shape[0]:
            raise ValueError(
                "Input vector length must match the covariance matrix size."
            )
        np.random.seed(seed)
        n = len(x)
        R = np.random.rand(n, n)
        R = min(x) + (max(x) - min(x)) * R
        self.V = (R @ R.T) / n
        return self

    def simulate_autocorrelation(self, x: Union[None, npt.NDArray[1]] = None) -> Self:
        if x is None:
            x = np.diagonal(self.V)
        elif len(x) != self.V.shape[0]:
            raise ValueError(
                "Input vector length must match the covariance matrix size."
            )
        n = len(x)
        for i in range(n):
            for j in range(i, n):
                lag = abs(i - j)
                # a = np.correlate(x[i:], np.roll(x[j:], lag), mode="valid")[0] / n
                x1 = x[0 : (n - lag)]
                x2 = np.roll(x, lag)[0 : (n - lag)]
                a = np.correlate(x1, x2, mode="valid")[0] / n
                self.V[i, j] = a
                self.V[j, i] = a
        return self

    def simulate_kinship(
        self,
        x: Union[None, npt.NDArray[1]] = None,
        k: int = 3,  # Number of kinship groups
        r: float = 0.25,  # Correlation strength within groups
        eps: float = 0.01,  # Small random noise to add
        seed: int = 42,
    ) -> Self:
        if x is None:
            x = np.diagonal(self.V)
        elif len(x) != self.V.shape[0]:
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
        func: Callable = CovarianceMatrix.simulate_simple,
        dmeans: Callable = [norm, uniform, laplace, beta][0],
        pmeans: tuple[float, float] = (2.0, 1.0),
        label_prefix: str = "param",
        seed: int = 42,
    ) -> None:
        K = CovarianceMatrix(p=p, label_prefix=label_prefix)
        x = dmeans(pmeans[0], pmeans[1]).rvs(size=p)
        if func == CovarianceMatrix.simulate_spherical:
            func(K, pmeans[0])
        else:
            func(K, x)
        if np.abs(np.linalg.det(K.V)) <= 1e-7:
            K.inflate_diagonal(s=0.01, tol=1e-7, max_iter=10)
        MVN = multivariate_normal(mean=x, cov=K.V, allow_singular=False, seed=seed)
        self.K = K
        self.b = MVN.rvs(size=1)
        self.p = len(self.b)
        return None

    def __repr__(self):
        print("Covariances:")
        print(self.K)
        print("Effects:")
        with np.printoptions(precision=2, suppress=True):
            print("b:", self.b)
        return "Effects(p={})".format(self.p)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Effects):
            return NotImplemented
        return (self.K == other.K) and np.array_equal(self.b, other.b)

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, Effects):
            return NotImplemented
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((hash(self.K), tuple(self.b.flatten())))


def expand_grid(*vec_effects: npt.NDArray[1]) -> npt.NDArray[2]:
    n = len(vec_effects)
    p = functools.reduce(lambda x, y: x * y, map(len, vec_effects), 1)
    out = np.array(np.meshgrid(*vec_effects)).reshape(n, p).T
    return out


input = {
    "years": [5, norm, (0.5, 1.0)],
    "sites": [3, norm, (1.0, 1.0)],
    "treatments": [4, laplace, (2.0, 4.0)],
    "entries": [10, beta, (2.0, 5.0)],
    "replications": [3, norm, (0.0, 1.0)],
    "rows": [6, beta, (2.0, 5.0)],
    "cols": [5, beta, (5.0, 2.0)],
    "residuals": [5 * 3 * 4 * 10 * 3, norm, (0.0, 1.0)],
    "sites:treatments": [12, norm, (2.0, 5.0)],
}
n_entries = input["entries"][0]
n_replications = input["replications"][0]
n_rows = input["rows"][0]
n_cols = input["cols"][0]
n_total = input["residuals"][0]

n_trials = functools.reduce(
    lambda x, y: x * y,
    [
        v[0]
        for k, v in input.items()
        if (k not in ["entries", "replications", "rows", "cols", "residuals"])
        and (len(k.split(":")) == 1)
    ],
    1,
)
n_per_trial = n_entries * n_replications
n_plots_per_trial = n_rows * n_cols

if (n_trials * n_per_trial) != n_total:
    raise ValueError(
        "Total number of data points is not equal to the number of residual effects."
    )

if n_per_trial != n_plots_per_trial:
    raise ValueError(
        "Number of entries multiplied by replications must equal number of rows multiplied by cols."
    )

effects = [
    Effects(p=v[0], dmeans=v[1], pmeans=v[2], label_prefix=k) for k, v in input.items()
]

effects_no_spat = [
    e
    for e in effects
    if (e.K.labels[0].split("-")[0] != "rows")
    and (e.K.labels[0].split("-")[0] != "cols")
]
vec_effects_no_spat = [np.array(e.b) for e in effects_no_spat]

effects_no_entrep = [
    e
    for e in effects
    if (e.K.labels[0].split("-")[0] != "entries")
    and (e.K.labels[0].split("-")[0] != "replications")
]
vec_effects_no_entrep = [np.array(e.b) for e in effects_no_entrep]
idx_rows_cols = [
    i
    for i, e in enumerate(effects_no_entrep)
    if (e.K.labels[0].split("-")[0] == "rows")
    or (e.K.labels[0].split("-")[0] == "cols")
]


p = functools.reduce(lambda a, b: a * b, [e.p for e in effects_no_spat], 1)
error = Effects(p=p, func=CovarianceMatrix.simulate_spherical, pmeans=(1.0, 0.0))


A = np.column_stack(
    (
        expand_grid(*vec_effects_no_spat),
        expand_grid(*vec_effects_no_entrep)[:, idx_rows_cols],
        error.b,
    )
)
