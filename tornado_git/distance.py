import numpy as np

def _random_directions(d: int, n_proj: int, rng: np.random.Generator) -> np.ndarray:
    """(d, n_proj) unit directions."""
    D = rng.normal(size=(d, n_proj))
    D /= (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    return D

def w2_1d_squared_exact(u: np.ndarray, v: np.ndarray) -> float:
    """
    Exact 1D W2^2 between two empirical measures with uniform weights.
    Works for unequal sample sizes.

    Computes: ∫_0^1 (Q_u(t) - Q_v(t))^2 dt
    where Q_u, Q_v are empirical quantile functions.
    """
    u = np.sort(np.asarray(u).ravel())
    v = np.sort(np.asarray(v).ravel())
    n, m = u.size, v.size
    if n == 0 or m == 0:
        raise ValueError("Empty input to w2_1d_squared_exact.")

    i = j = 0
    t = 0.0
    w2 = 0.0

    # breakpoints at (i+1)/n and (j+1)/m
    inv_n = 1.0 / n
    inv_m = 1.0 / m

    while i < n and j < m:
        next_t_u = (i + 1) * inv_n
        next_t_v = (j + 1) * inv_m
        next_t = next_t_u if next_t_u < next_t_v else next_t_v

        dt = next_t - t
        diff = u[i] - v[j]
        w2 += dt * (diff * diff)

        t = next_t
        if next_t == next_t_u:
            i += 1
        if next_t == next_t_v:
            j += 1

    # t should be 1.0 here (up to floating error)
    return w2

def sliced_wasserstein_2(
    X: np.ndarray,
    Y: np.ndarray,
    n_proj: int = 400,
    seed: int = 0,
    standardize: bool = True,
    chunk_size: int = 64,
) -> float:
    """
    Sliced Wasserstein distance SW2 between X (N,d) and Y (M,d):
      SW2 = sqrt( mean_l W2^2( <X,theta_l>, <Y,theta_l> ) )

    standardize=True z-scores features using stats from X ∪ Y.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must be 2D with the same second dimension.")
    d = X.shape[1]

    if standardize:
        Z = np.vstack([X, Y])
        mu = Z.mean(axis=0, keepdims=True)
        sd = Z.std(axis=0, ddof=0, keepdims=True)
        sd = np.where(sd < 1e-12, 1.0, sd)
        X = (X - mu) / sd
        Y = (Y - mu) / sd

    rng = np.random.default_rng(seed)
    dirs = _random_directions(d, n_proj, rng)

    total_w2 = 0.0
    done = 0
    while done < n_proj:
        k = min(chunk_size, n_proj - done)
        D = dirs[:, done:done + k]   # (d, k)
        Xp = X @ D                   # (N, k)
        Yp = Y @ D                   # (M, k)

        for j in range(k):
            total_w2 += w2_1d_squared_exact(Xp[:, j], Yp[:, j])

        done += k

    return np.sqrt(total_w2 / n_proj)





def mean_cov_distance(X, Y ):
    """
    Parameters
    ----------
    X : array-like, shape (N, dim)
    Y : array-like, shape (M, dim)

    Returns
    -------
    float
        ||mu_X - mu_Y||_2^2 / dim + ||Sigma_X - Sigma_Y||_F^2 / dim^2

    where
        mu_X = (1/N) sum_i X_i,
        Sigma_X = (1/N) sum_i (X_i - mu_X)(X_i - mu_X)^T,
    and similarly for Y.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)


    N, dim = X.shape
    M, _ = Y.shape

    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)

    Xc = X - mu_X
    Yc = Y - mu_Y

    Sigma_X = (Xc.T @ Xc) / N
    Sigma_Y = (Yc.T @ Yc) / M

    mean_term = np.sum((mu_X - mu_Y) ** 2) / dim
    cov_term = np.sum((Sigma_X - Sigma_Y) ** 2) / (dim ** 2)

    return np.sqrt(mean_term + cov_term)



import ot

def wasserstein_distance_point_clouds(
    data1,
    data2,
    p=2,
    weights1=None,
    weights2=None,
    sinkhorn=False,
    reg=1e-2,
    standardize=False,
    return_plan=False
):
    """
    Compute empirical p-Wasserstein distance between two point clouds.

    Parameters
    ----------
    data1 : array-like, shape (N, d)
    data2 : array-like, shape (M, d)
    p : int or float, default=2
        Order of Wasserstein distance.
    weights1 : array-like, shape (N,), optional
        Weights on data1. If None, use uniform weights 1/N.
    weights2 : array-like, shape (M,), optional
        Weights on data2. If None, use uniform weights 1/M.
    sinkhorn : bool, default=False
        If True, compute entropically regularized OT.
        If False, compute exact OT.
    reg : float, default=1e-2
        Sinkhorn regularization parameter.
    standardize : bool, default=False
        If True, standardize coordinates using pooled mean/std
        before computing the distance.
    return_plan : bool, default=False
        If True, also return the transport plan.

    Returns
    -------
    Wp : float
        Empirical p-Wasserstein distance.
    plan : ndarray, optional
        Optimal transport plan if return_plan=True.
    """
    X = np.asarray(data1, dtype=float)
    Y = np.asarray(data2, dtype=float)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("data1 and data2 must both be 2D arrays.")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("data1 and data2 must have the same dimension.")

    N = X.shape[0]
    M = Y.shape[0]

    if weights1 is None:
        a = np.ones(N) / N
    else:
        a = np.asarray(weights1, dtype=float)
        a = a / a.sum()

    if weights2 is None:
        b = np.ones(M) / M
    else:
        b = np.asarray(weights2, dtype=float)
        b = b / b.sum()

    if standardize:
        Z = np.vstack([X, Y])
        mu = Z.mean(axis=0)
        sd = Z.std(axis=0, ddof=0)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd
        Y = (Y - mu) / sd

    # Cost matrix C_{ij} = ||x_i - y_j||^p
    C = ot.dist(X, Y, metric='euclidean') ** p

    if sinkhorn:
        if return_plan:
            plan = ot.sinkhorn(a, b, C, reg=reg)
            cost = np.sum(plan * C)
            return cost ** (1.0 / p), plan
        else:
            cost = ot.sinkhorn2(a, b, C, reg=reg)
            return cost ** (1.0 / p)
    else:
        if return_plan:
            plan = ot.emd(a, b, C)
            cost = np.sum(plan * C)
            return cost ** (1.0 / p), plan
        else:
            cost = ot.emd2(a, b, C)
            return cost ** (1.0 / p)


