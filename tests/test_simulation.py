import numpy as np
from scripts.py.simulate import CovarianceMatrix


def test_CovarianceMatrix():
    # Test basic initialization with p=3
    p = 3
    C = CovarianceMatrix(p=p)
    assert C.V.shape == (3, 3)

    # Test initialization with p=5 and verify default labels
    p = 5
    C = CovarianceMatrix(p=p)
    assert C.V.shape == (p, p)
    assert C.labels == [f"param-{i + 1}" for i in range(5)]
    
    # Test large matrix initialization with custom label prefix
    p = 10_000
    C = CovarianceMatrix(p=p, label_prefix="test")
    assert C.V.shape == (p, p)
    assert C.labels == [f"test-{str(i + 1).zfill(5)}" for i in range(p)]

    # Test spherical covariance simulation
    p = 3
    C = CovarianceMatrix(p=p)
    C.simulate_spherical(s=np.pi)
    assert C.V.diagonal().sum() == p * np.pi

    # Test simple covariance simulation using outer product
    p = 5
    C = CovarianceMatrix(p=p)
    x = np.array(range(p))
    C.simulate_simple(x = x)
    assert (C.V == np.outer(x, x) / p).all()

    # Test diagonal covariance simulation
    p = 10_000
    C = CovarianceMatrix(p=p)
    x = np.random.rand(p)
    C.simulate_diagonal(x=x)
    assert min(C.V.diagonal()) >= min(x)
    assert max(C.V.diagonal()) <= max(x)

    # Test random covariance simulation with bounds verification
    p = 3
    C = CovarianceMatrix(p=p)
    x = np.array(range(p))
    C.simulate_random(x=x)
    assert min(C.V.flatten()) >= min([0.0, -min(x), min(x)**2])
    assert max(C.V.flatten()) <= p*(max(x)**2)

    # Test autocorrelation covariance simulation
    p = 5
    C = CovarianceMatrix(p=p)
    x = np.random.rand(p)
    C.simulate_autocorrelation(x=x)
    # Verify minimum correlation at maximum distance
    assert C.V[p-1, 0] == C.V[0, p-1] == min(C.V.flatten())
    # Verify diagonal dominance
    assert ([C.V[i, i] == max(C.V[i, :]) for i in range(p)] == np.repeat(np.True_, p)).all()
    assert ([C.V[j, j] == max(C.V[:, j]) for j in range(p)] == np.repeat(np.True_, p)).all()
    # Verify diagonal scaling, i.e. common variance across the diagonal
    assert C.V[0, 0] == C.V.diagonal().sum() / p
    
    np.outer(x, x.T) @ C.V @ np.outer(x, x.T)
    