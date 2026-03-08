import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import Legendre
from numpy.polynomial.polynomial import polyval


def shifted_legendre_coef(n, normalized=True):
    """
    Return Coef of shape (n, n), where row k contains the monomial
    coefficients of the k-th shifted Legendre basis polynomial on [0,1].

    If normalized=True, basis_k(x) = sqrt(2k+1) * P_k(2x-1).
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    Coef = np.zeros((n, n), dtype=float)

    x_poly = Polynomial([0.0, 1.0])   # x
    t_poly = 2.0 * x_poly - 1.0       # 2x - 1

    for k in range(n):
        Pk_t = Legendre.basis(k).convert(kind=Polynomial)   # P_k(t)
        fk_x = Pk_t(t_poly)                                 # P_k(2x-1)

        if normalized:
            fk_x = np.sqrt(2 * k + 1) * fk_x

        coef = fk_x.coef
        Coef[k, :len(coef)] = coef

    return Coef


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

    Here g is represented through shifted Legendre basis coefficients,
    but internally converted to monomial coefficients in x.
    """

    def __init__(self, n: int, normalized=True, seed=None):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = int(n)
        self.normalized = normalized
        self._rng = np.random.default_rng(seed)

        # coef_f[k] = monomial coefficients of basis function f_k(x)
        self.coef_f = shifted_legendre_coef(self.n, normalized=self.normalized)  # (n, n)

        # coef_F[k] = monomial coefficients of integral_0^x f_k(t) dt
        self.coef_F = np.zeros((self.n, self.n + 1), dtype=float)
        self.coef_F[:, 1:] = self.coef_f / np.arange(1, self.n + 1, dtype=float)

        self._grid_cache = {}

    def _density_poly(self, a):
        """
        Return monomial coefficients c of
            g(x) = sum_{j=0}^{n-1} c[j] x^j
        """
        a = np.asarray(a, dtype=float).reshape(-1)
        if a.shape[0] != self.n:
            raise ValueError(f"a must have length {self.n}")
        return a @ self.coef_f

    def _mass_poly(self, a):
        """
        Return monomial coefficients of
            G(x) = integral_0^x g(t) dt
        and Z = G(1).
        """
        a = np.asarray(a, dtype=float).reshape(-1)
        if a.shape[0] != self.n:
            raise ValueError(f"a must have length {self.n}")

        coef_G = a @ self.coef_F
        Z = float(polyval(1.0, coef_G))
        return coef_G, Z

    def f(self, a, x):
        """
        Evaluate g(x).
        """
        coef_g = self._density_poly(a)
        return polyval(x, coef_g)

    def F(self, a, x, eps_mass=1e-14):
        """
        Evaluate normalized CDF
            F(x) = G(x) / G(1),
        where G(x) = integral_0^x g(t) dt.
        """
        coef_G, Z = self._mass_poly(a)
        if not np.isfinite(Z) or Z <= eps_mass:
            raise ValueError(f"Mass Z must be positive and finite. Got Z={Z}.")
        return polyval(x, coef_G) / Z

    def _get_grid(self, grid_size, anchor_weight):
        key = (int(grid_size), float(anchor_weight))
        if key in self._grid_cache:
            return self._grid_cache[key]

        grid_size = key[0]
        if grid_size < 16:
            raise ValueError("grid_size should be at least 16 for stability.")

        xg = np.linspace(0.0, 1.0, grid_size, dtype=float)
        w = np.ones(grid_size, dtype=float)
        w[0] = w[-1] = key[1]
        self._grid_cache[key] = (xg, w)
        return xg, w

    def _inv_cdf_isotonic(self, u, coef_G, Z, grid_size, anchor_weight):
        if not np.isfinite(Z) or Z <= 0.0:
            raise ValueError(f"Mass Z must be positive and finite. Got Z={Z}.")

        u = np.asarray(u, dtype=float)
        if np.any(u < 0.0) or np.any(u > 1.0):
            raise ValueError("u must be in [0,1].")

        xg, w = self._get_grid(grid_size, anchor_weight)

        Fg = polyval(xg, coef_G) / Z
        Fg = np.asarray(Fg, dtype=float)
        Fg[0] = 0.0
        Fg[-1] = 1.0
        Fg = np.clip(Fg, 0.0, 1.0)

        F_iso = _pava_increasing(Fg, w=w)

        lo, hi = float(F_iso[0]), float(F_iso[-1])
        if hi <= lo + 1e-15:
            raise ValueError("Degenerate isotonic CDF after repair.")

        F_iso = (F_iso - lo) / (hi - lo)
        F_iso[0] = 0.0
        F_iso[-1] = 1.0
        F_iso = np.maximum.accumulate(F_iso)

        F_unique, idx = np.unique(F_iso, return_index=True)
        x_unique = xg[idx]
        if F_unique.size < 2:
            raise ValueError("CDF grid collapsed after isotonic projection.")

        return np.interp(u, F_unique, x_unique)

    def sample(self, a, size=1, grid_size=1024, anchor_weight=1e6, eps_mass=1e-9):
        """
        Draw samples from g(x) / integral_0^1 g using isotonic-regression inversion.
        """
        a = np.asarray(a, dtype=float).reshape(-1).copy()
        if a.shape[0] != self.n:
            raise ValueError(f"a must have length {self.n}")

        size = int(size)
        if size < 1:
            raise ValueError("size must be >= 1")

        coef_G, Z = self._mass_poly(a)
        if Z <= eps_mass:
            return -np.inf

        u = self._rng.random(size)
        x = self._inv_cdf_isotonic(
            u, coef_G, Z, grid_size=grid_size, anchor_weight=anchor_weight
        )

        return float(x[0]) if size == 1 else x