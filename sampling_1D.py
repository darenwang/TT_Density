import numpy as np
from numpy.polynomial.legendre import leg2poly
from numpy.polynomial.polynomial import polyval


def _pava_increasing(y, w):
    """
    Weighted isotonic regression (increasing) via PAVA.
    Minimizes sum_i w_i (y_i - yhat_i)^2 subject to yhat nondecreasing.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    n = y.size
    if w.shape != y.shape:
        raise ValueError("w must have the same shape as y")

    starts = np.empty(n, dtype=int)
    ends = np.empty(n, dtype=int)
    ws = np.empty(n, dtype=float)
    ms = np.empty(n, dtype=float)

    m = 0
    for i in range(n):
        starts[m] = i
        ends[m] = i
        ws[m] = w[i]
        ms[m] = y[i]

        while m > 0 and ms[m - 1] > ms[m]:
            w_new = ws[m - 1] + ws[m]
            m_new = (ws[m - 1] * ms[m - 1] + ws[m] * ms[m]) / w_new
            ws[m - 1] = w_new
            ms[m - 1] = m_new
            ends[m - 1] = ends[m]
            m -= 1

        m += 1

    y_hat = np.empty(n, dtype=float)
    for j in range(m):
        y_hat[starts[j] : ends[j] + 1] = ms[j]
    return y_hat


class Legendre_Sampler:
    """
    Sample on [0,1] from g(x)/∫_0^1 g using:
      1) build F(x)=G(x)/Z on a grid
      2) isotonic regression to enforce monotone CDF
      3) invert by interpolation

    Randomness:
      - internal np.random.default_rng(), no seed specified
      - no rng parameter exposed
    """

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = int(n)
        self._rng = np.random.default_rng()

        # Build coef_f for f_k(x) = sqrt(2k+1) P_k(t), as monomials in t=2x-1
        coef_leg = np.zeros((self.n, self.n), dtype=float)
        for k in range(self.n):
            c = np.zeros(k + 1, dtype=float)
            c[-1] = 1.0
            pk = leg2poly(c)                      # P_k(t) in monomials
            coef_leg[k, : pk.shape[0]] = pk

        scale = np.sqrt(2.0 * np.arange(self.n, dtype=float) + 1.0)
        coef_f = coef_leg * scale[:, None]        # (n, n)

        # Build coef_F for F_k(x)=∫_0^x f_k(u)du, as monomials in t
        coef_F = np.zeros((self.n, self.n + 1), dtype=float)
        denom = 2.0 * (np.arange(self.n, dtype=float) + 1.0)  # 2(m+1)
        coef_F[:, 1:] = coef_f / denom[None, :]

        # enforce F_k(0)=0: at x=0, t=-1 so t^m = (-1)^m
        signs = (-1.0) ** np.arange(self.n + 1, dtype=float)
        coef_F[:, 0] -= coef_F @ signs

        self.coef_F = coef_F                      # (n, n+1)
        self._grid_cache = {}                     # optional speed: cache (xg,tg,w)

    def _mass_poly(self, a):
        a = np.asarray(a, dtype=float).reshape(-1)
        if a.shape[0] != self.n:
            raise ValueError(f"a must have length {self.n}")
        coef_G = a @ self.coef_F                  # (n+1,)
        Z = float(polyval(1.0, coef_G))           # x=1 => t=1
        return coef_G, Z

    def _get_grid(self, grid_size, anchor_weight):
        key = (int(grid_size), float(anchor_weight))
        if key in self._grid_cache:
            return self._grid_cache[key]

        grid_size = key[0]
        if grid_size < 16:
            raise ValueError("grid_size should be at least 16 for stability.")

        xg = np.linspace(0.0, 1.0, grid_size, dtype=float)
        tg = 2.0 * xg - 1.0
        w = np.ones(grid_size, dtype=float)
        w[0] = w[-1] = key[1]
        self._grid_cache[key] = (xg, tg, w)
        return xg, tg, w

    def _inv_cdf_isotonic(self, u, coef_G, Z, grid_size, anchor_weight):
        
        #########pay attension here
        #########floating points
        if not np.isfinite(Z) or Z <= 0.0:
            
            raise ValueError(f"Mass Z must be positive and finite. Got Z={Z}.")

        u = np.asarray(u, dtype=float)
        if np.any(u < 0.0) or np.any(u > 1.0):
            raise ValueError("u must be in [0,1].")

        xg, tg, w = self._get_grid(grid_size, anchor_weight)

        Fg = polyval(tg, coef_G) / Z
        Fg = np.asarray(Fg, dtype=float)
        Fg[0] = 0.0
        Fg[-1] = 1.0
        Fg = np.clip(Fg, 0.0, 1.0)

        F_iso = _pava_increasing(Fg, w=w)

        # normalize to [0,1] exactly
        lo, hi = float(F_iso[0]), float(F_iso[-1])
        if hi <= lo + 1e-15:
            return u.copy()

        F_iso = (F_iso - lo) / (hi - lo)
        F_iso[0] = 0.0
        F_iso[-1] = 1.0
        F_iso = np.maximum.accumulate(F_iso)

        # remove duplicates for interp stability
        F_unique, idx = np.unique(F_iso, return_index=True)
        x_unique = xg[idx]
        if F_unique.size < 2:
            return u.copy()

        return np.interp(u, F_unique, x_unique)

    def sample(self, a, size=1, grid_size=1024, anchor_weight=1e6, eps_mass=10**-9):
        """
        Draw samples from g(x)/∫ g using isotonic-regression inversion.
        """
        a = np.asarray(a, dtype=float).reshape(-1).copy()
        if a.shape[0] != self.n:
            raise ValueError(f"a must have length {self.n}")

        coef_G, Z = self._mass_poly(a)
        if Z <= eps_mass:
            return -np.inf
            #a[0] = max(a[0], eps_mass)           # minimal mass repair
            #coef_G, Z = self._mass_poly(a)

        u = self._rng.random(int(size))
        return self._inv_cdf_isotonic(u, coef_G, Z, grid_size=grid_size, anchor_weight=anchor_weight)[0]
