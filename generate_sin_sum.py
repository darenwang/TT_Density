import numpy as np


def _sample_one_coordinate_ar(aa: float, A: float, phi: float) -> float:
    """
    Sample x in [0,1] with unnormalized density proportional to:
        1 + A * sin(aa*x + phi),
    using Uniform(0,1) proposal and constant envelope (1 + |A|).
    """
    M = 1.0 + abs(A)  # valid envelope height for any sign of A
    while True:
        u = np.random.random()
        v = np.random.random()
        if v * M <= 1.0 + A * np.sin(aa * u + phi):
            return u


def sample_sin_sum(aa: float, dim: int, N_sample: int) -> np.ndarray:
    """
    Exact sampler on [0,1]^dim from density proportional to
        f(x) = sin(aa * (x1 + ... + xd)) + 1,
    using sequential 1D accept-reject on closed-form conditionals.

    Inputs: aa>0, dim>=1, N_sample>=1
    Output: X of shape (N_sample, dim)
    """
    aa = float(aa)
    if aa <= 0:
        raise ValueError("aa must be > 0.")
    dim = int(dim)
    if dim < 1:
        raise ValueError("dim must be >= 1.")
    N_sample = int(N_sample)
    if N_sample < 1:
        raise ValueError("N_sample must be >= 1.")

    c = 2.0 * np.sin(0.5 * aa) / aa  # can be negative for some aa

    X = np.empty((N_sample, dim), dtype=float)
    for i in range(N_sample):
        s = 0.0  # partial sum
        for k in range(dim):
            m = dim - k - 1  # remaining coords after x_k
            A = c ** m
            phi = aa * s + 0.5 * m * aa
            xk = _sample_one_coordinate_ar(aa, A, phi)
            X[i, k] = xk
            s += xk

    return X


def conditional_sample_sin_sum(
    aa: float,
    dim: int,
    x_given,
    N_sample: int,
    return_full: bool = False,
) -> np.ndarray:
    """
    Conditional sampler under f(x) = sin(aa * sum_i x_i) + 1 on [0,1]^dim.

    We condition on the FIRST s coordinates being fixed to x_given (length s, with s < dim),
    and sample the remaining dim-s coordinates from the exact conditional.

    Inputs
    ------
    aa : float, > 0
    dim : int, total dimension
    x_given : array-like, shape (s,), values in [0,1], with 0 <= s < dim
    N_sample : int
    return_full : if True, return (N_sample, dim) with x_given prepended;
                  if False, return only the sampled tail (N_sample, dim-s).

    Output
    ------
    tail or full samples as described above.
    """
    aa = float(aa)
    if aa <= 0:
        raise ValueError("aa must be > 0.")
    dim = int(dim)
    if dim < 1:
        raise ValueError("dim must be >= 1.")
    N_sample = int(N_sample)
    if N_sample < 1:
        raise ValueError("N_sample must be >= 1.")

    x_given = np.asarray(x_given, dtype=float).reshape(-1)
    s_given = int(x_given.size)
    if not (0 <= s_given < dim):
        raise ValueError("x_given must have length s with 0 <= s < dim.")
    if np.any(x_given < 0.0) or np.any(x_given > 1.0):
        raise ValueError("All entries of x_given must be in [0,1].")

    c = 2.0 * np.sin(0.5 * aa) / aa

    tail_dim = dim - s_given
    tail = np.empty((N_sample, tail_dim), dtype=float)

    s0 = float(x_given.sum())

    for i in range(N_sample):
        s = s0
        for j in range(tail_dim):
            k_global = s_given + j
            m = dim - k_global - 1
            A = c ** m
            phi = aa * s + 0.5 * m * aa
            xk = _sample_one_coordinate_ar(aa, A, phi)
            tail[i, j] = xk
            s += xk

    if not return_full:
        return tail

    full = np.empty((N_sample, dim), dtype=float)
    if s_given > 0:
        full[:, :s_given] = x_given[None, :]
    full[:, s_given:] = tail
    return full



def sin_sum_density(aa: float, dim: int, x_input, normalized: bool = True):
    """
    f(x) = sin(aa * sum(x)) + 1 on [0,1]^dim.
    If normalized=True, returns p(x) = f(x)/Z where Z is the exact integral over the cube.

    x_input can be shape (dim,) or (N, dim).
    """
    aa = float(aa)
    dim = int(dim)
    x = np.asarray(x_input, dtype=float)

    if x.ndim == 1:
        if x.shape[0] != dim:
            raise ValueError(f"x_input has length {x.shape[0]}, expected dim={dim}.")
        x2 = x[None, :]
        squeeze = True
    elif x.ndim == 2:
        if x.shape[1] != dim:
            raise ValueError(f"x_input has shape {x.shape}, expected (*, {dim}).")
        x2 = x
        squeeze = False
    else:
        raise ValueError("x_input must have shape (dim,) or (N, dim).")

    if np.any(x2 < 0.0) or np.any(x2 > 1.0):
        raise ValueError("x_input must lie in [0,1]^dim.")

    s = np.sum(x2, axis=1)
    f = np.sin(aa * s) + 1.0

    if not normalized:
        return f[0] if squeeze else f

    # c = 2*sin(aa/2)/aa, with a tiny-aa safeguard
    if abs(aa) < 1e-12:
        c = 1.0
    else:
        c = 2.0 * np.sin(aa / 2.0) / aa

    Z = 1.0 + (c ** dim) * np.sin(aa * dim / 2.0)
    p = f / Z

    return p[0] if squeeze else p