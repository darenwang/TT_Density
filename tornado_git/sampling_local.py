import numpy as np


class Legendre_Sampler:
    """
    # Accept-reject sampler on [0,1] for the basis
    #     {1, phi_0, ..., phi_{n-2}},
    # where:
    #   - 1 denotes the function equal to 1 on [0,1] and 0 outside [0,1];
    #   - phi_0(x) = 1 - (n-1)x on [0, 1/(n-1)], and 0 otherwise;
    #   - for i = 1, ..., n-2, phi_i is the usual hat function on the mesh
    #     x_j = j/(n-1), namely
    #
    #         phi_i(x) = (x - x_{i-1}) / (x_i - x_{i-1})   for x in [x_{i-1}, x_i],
    #                  = (x_{i+1} - x) / (x_{i+1} - x_i)   for x in [x_i, x_{i+1}],
    #                  = 0                                  otherwise.

    Input vector a has length n and represents
        g(x) = a[0] * 1_[0,1](x) + sum_{i=0}^{n-2} a[i+1] * phi_i(x).

    The sampler returns draws from a density proportional to g^+(x),
    where g^+(x) = max(g(x), 0).
    """

    def __init__(self, n: int,L, normalized=True, seed=None):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.L = L
        self.n = int(n)
        self.normalized = normalized   # kept only for compatibility
        self._rng = np.random.default_rng(seed)

        if self.n >= 2:
            self.num_intervals = self.n - 1
            self.h = 1.0 / self.num_intervals
        else:
            self.num_intervals = 0
            self.h = None

    def f(self, a, x):
        """
        Evaluate
            g(x) = a[0] + sum_{i=0}^{n-2} a[i+1] phi_i(x)
        on [0,1], with value 0 outside [0,1].
        """
        a = np.asarray(a, dtype=float).reshape(-1)
        if a.shape[0] != self.n:
            raise ValueError(f"a must have length {self.n}")

        x = np.asarray(x, dtype=float)
        scalar_input = (x.ndim == 0)
        x_flat = x.reshape(-1)

        out = np.zeros_like(x_flat, dtype=float)

        # Restrict to [0,1]
        inside = (x_flat >= 0.0) & (x_flat <= 1.0)
        if not np.any(inside):
            return float(out[0]) if scalar_input else out.reshape(x.shape)

        xi = x_flat[inside]

        # Special case n = 1: basis is just {1}
        if self.n == 1:
            out[inside] = a[0]
            return float(out[0]) if scalar_input else out.reshape(x.shape)

        m = self.num_intervals
        h = self.h

        # Interval index k so that xi in [x_k, x_{k+1}]
        # For xi == 1, force k = m-1
        k = np.minimum((xi / h).astype(int), m - 1)

        xk = k * h
        t = (xi - xk) / h   # local coordinate in [0,1]

        vals = np.full_like(xi, a[0], dtype=float)

        # Contribution from phi_k
        vals += a[k + 1] * (1.0 - t)

        # Contribution from phi_{k+1}, except on the last interval
        mask_next = (k + 1 <= m - 1)
        if np.any(mask_next):
            vals[mask_next] += a[k[mask_next] + 2] * t[mask_next]

        out[inside] = vals
        return float(out[0]) if scalar_input else out.reshape(x.shape)

    def _max_value(self, a):
        """
        Exact maximum of g on [0,1].
        """
        a = np.asarray(a, dtype=float).reshape(-1)
        if a.shape[0] != self.n:
            raise ValueError(f"a must have length {self.n}")

        if self.n == 1:
            return float(a[0])

        return float(a[0] + max(0.0, np.max(a[1:])))

    def sample(self, a, size=1, upper_bound_threshold=1e-9):
        """
        Draw samples from a density proportional to g^+(x) on [0,1],
        where g^+(x) = max(g(x), 0).

        Uses uniform proposal on [0,1] and the exact bound
            M = max_{x in [0,1]} g(x).

        If M <= upper_bound_threshold, return -np.inf.
        """
        a = np.asarray(a, dtype=float).reshape(-1)@self.L.T
        if a.shape[0] != self.n:
            raise ValueError(f"a must have length {self.n}")

        size = int(size)
        if size < 1:
            raise ValueError("size must be >= 1")

        M = self._max_value(a)
        if (not np.isfinite(M)) or (M <= upper_bound_threshold):
            return -np.inf

        out = np.empty(size, dtype=float)
        filled = 0

        while filled < size:
            batch = max(1024, 4 * (size - filled))

            x = self._rng.random(batch)         # Uniform[0,1]
            gx = np.asarray(self.f(a, x), dtype=float)
            g_plus = np.clip(gx, 0.0, np.inf)

            u = self._rng.random(batch)
            keep = x[u <= (g_plus / M)]

            n_take = min(size - filled, keep.size)
            if n_take > 0:
                out[filled:filled + n_take] = keep[:n_take]
                filled += n_take

        return float(out[0]) if size == 1 else out