import numpy as np
from scipy.stats import norm, uniform, laplace, beta


def simulate_spherical_covariance_matrix(
    y: np.ndarray[1], s: float = 1.00
) -> np.ndarray[2]:
    n = len(y)
    return np.eye(n, n) * s


def simulate_simple_covariance_matrix(y: np.ndarray[1]) -> np.ndarray[2]:
    return np.outer(y, y)


def simulate_diagonal_covariance_matrix(
    y: np.ndarray[1], seed: int = 42
) -> np.ndarray[2]:
    np.random.seed(seed)
    n = len(y)
    diagonal_elements = np.random.uniform(0.1, 1.0, size=n)
    return np.diag(diagonal_elements)


def simulate_random_covariance_matrix(
    y: np.ndarray[1], seed: int = 42
) -> np.ndarray[2]:
    np.random.seed(seed)
    n = len(y)
    R = np.random.rand(n, n)
    K = R @ R.T
    return K


def simulate_autocorrelation_covariance_matrix(y: np.ndarray[1]) -> np.ndarray[2]:
    n = len(y)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            lag = abs(i - j)
            if lag < n:
                K[i, j] = np.correlate(y, np.roll(y, lag), mode="valid")[0]
    return K


def simulate_kinship_covariance_matrix(
    y: np.ndarray[1], k: int = 3, r: float = 0.25, eps: float = 0.01, seed: int = 42
) -> np.ndarray[2]:
    n = len(y)
    if n < k:
        raise ValueError("Input vector must have at least k elements.")
    np.random.seed(seed)
    groupings = np.repeat(np.arange(k), np.ceil(n / k))[:n]
    np.random.shuffle(groupings)
    K = np.zeros((n, n))
    for i in range(k):
        (idx,) = np.where(groupings == i)
        row_indices, col_indices = np.meshgrid(idx, idx)
        K[row_indices, col_indices] += r
    K += np.random.normal(0, eps, (n, n))
    return K


def simulate_entry_effects(
    n: int = 100,
    distribution=[norm, uniform, laplace, beta][0],
    dparameters: tuple[float, float] = (2.0, 1.0),
) -> np.array[1]:
    # n=100; distribution=[norm, uniform, laplace, beta][1]; dparameters = (2.0, 1.0)
    D = distribution(dparameters[0], dparameters[1])
    y = D.rvs(size=n)
    K = simulate_autocorrelation_matrix(y)
    K.shape

    return y


def simulate_location_effects(
    n: int = 20,
    distribution=[norm, uniform, laplace, beta][0],
    dparameters: tuple[float, float] = (2.0, 1.0),
) -> np.array[1]:
    # s=20; distribution=[norm, uniform, laplace, beta][1]; dparameters = (2.0, 1.0)
    D = distribution(dparameters[0], dparameters[1])
    return D.rvs(size=n)


def simulate_season_effects(
    n: int = 20,
    distribution=[norm, uniform, laplace, beta][0],
    dparameters: tuple[float, float] = (2.0, 1.0),
) -> np.array[1]:
    # s=20; distribution=[norm, uniform, laplace, beta][1]; dparameters = (2.0, 1.0)
    D = distribution(dparameters[0], dparameters[1])
    return D.rvs(size=n)
