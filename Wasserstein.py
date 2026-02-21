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
    n_proj: int = 200,
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
