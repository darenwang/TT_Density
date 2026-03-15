import math
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


class Legendre_Sampler:
    """
    Sampler on [0,1] using a Bernstein positive envelope.

    If
        g(x) = sum_j a[j] * phi_j(x),
    where phi_j are shifted Legendre basis functions, then this class:

      1) converts g into monomial coefficients,
      2) converts g into Bernstein coefficients b_k,
      3) builds h(x) = sum_k max(b_k, 0) * B_{k,m}(x),
      4) samples from normalized h as a Beta mixture,
      5) accepts with probability g^+(x) / h(x).

    Therefore sample(a, ...) returns draws from a density proportional to g^+.
    If g >= 0 on [0,1], this is exactly sampling from normalized g.
    """

    def __init__(self, n: int, normalized=True, seed=None):
        if n < 1:
            raise ValueError("n must be >= 1")

        self.n = int(n)
        self.degree = self.n - 1
        self.normalized = normalized
        self._rng = np.random.default_rng(seed)

        # coef_f[k] = monomial coefficients of basis function phi_k(x)
        self.coef_f = shifted_legendre_coef(self.n, normalized=self.normalized)

    def _density_poly(self, a):
        """
        Return monomial coefficients c of
            g(x) = sum_{j=0}^{n-1} c[j] x^j
        """
        a = np.asarray(a, dtype=float).reshape(-1)
        if a.shape[0] != self.n:
            raise ValueError(f"a must have length {self.n}")
        return a @ self.coef_f

    def f(self, a, x):
        """
        Evaluate g(x).
        """
        coef_g = self._density_poly(a)
        return polyval(x, coef_g)

    def _monomial_to_bernstein(self, c):
        """
        Convert monomial coefficients c of degree m polynomial
            p(x) = sum_{j=0}^m c[j] x^j
        into Bernstein coefficients b of degree m:
            p(x) = sum_{k=0}^m b[k] B_{k,m}(x)

        Formula:
            b[k] = sum_{j=0}^k c[j] * C(k,j) / C(m,j)
        """
        c = np.asarray(c, dtype=float).reshape(-1)
        m = c.size - 1
        b = np.zeros(m + 1, dtype=float)

        for k in range(m + 1):
            s = 0.0
            for j in range(k + 1):
                s += c[j] * math.comb(k, j) / math.comb(m, j)
            b[k] = s

        return b

    def bernstein_coef(self, a):
        """
        Return Bernstein coefficients of g on [0,1].
        """
        coef_g = self._density_poly(a)
        return self._monomial_to_bernstein(coef_g)

    def _bernstein_eval(self, b, x):
        """
        Evaluate a Bernstein polynomial
            p(x) = sum_k b[k] B_{k,m}(x)
        by de Casteljau's algorithm.
        """
        b = np.asarray(b, dtype=float).reshape(-1)
        x = np.asarray(x, dtype=float)

        scalar_input = (x.ndim == 0)
        x_flat = x.reshape(-1)

        temp = np.tile(b, (x_flat.size, 1))
        one_minus_x = 1.0 - x_flat
        m1 = b.size

        for r in range(1, m1):
            temp[:, :m1 - r] = (
                one_minus_x[:, None] * temp[:, :m1 - r]
                + x_flat[:, None] * temp[:, 1:m1 - r + 1]
            )

        out = temp[:, 0]
        if scalar_input:
            return float(out[0])
        return out.reshape(x.shape)

    def h(self, a, x):
        """
        Evaluate the Bernstein positive envelope
            h(x) = sum_k max(b_k, 0) B_{k,m}(x),
        where b_k are the Bernstein coefficients of g.
        """
        b = self.bernstein_coef(a)
        b_plus = np.clip(b, 0.0, np.inf)
        return self._bernstein_eval(b_plus, x)

    def sample(self, a, size=1, batch_size=None, upper_bound_threshold=1e-9):
        """
        Draw samples from a density proportional to g^+(x) on [0,1],
        where g^+(x) = max(g(x), 0).
    
        If the Bernstein upper bound
            M = max_k b_k^+
        is <= upper_bound_threshold, return -np.inf.
        """
        a = np.asarray(a, dtype=float).reshape(-1)
        if a.shape[0] != self.n:
            raise ValueError(f"a must have length {self.n}")
    
        size = int(size)
        if size < 1:
            raise ValueError("size must be >= 1")
    
        # Bernstein coefficients of g
        b = self.bernstein_coef(a)
    
        # Positive Bernstein envelope coefficients
        b_plus = np.clip(b, 0.0, np.inf)
    
        # Bernstein-based global upper bound of h, hence also of g^+
        M = float(np.max(b_plus))
        if (not np.isfinite(M)) or (M <= upper_bound_threshold):
            return -np.inf
    
        s = float(np.sum(b_plus))
        if (not np.isfinite(s)) or (s <= 0.0):
            return -np.inf
    
        # Proposal q is normalized h.
        # Since (m+1) B_{k,m}(x) is Beta(k+1, m-k+1) density,
        # q is a mixture of Beta(k+1, m-k+1) with weights proportional to b_plus[k].
        weights = b_plus / s
    
        out = np.empty(size, dtype=float)
        filled = 0
    
        while filled < size:
            m_batch = (
                max(1024, 4 * (size - filled))
                if batch_size is None
                else max(int(batch_size), size - filled)
            )
    
            # Sample proposal component
            k = self._rng.choice(self.degree + 1, size=m_batch, p=weights)
    
            # Sample from Beta(k+1, degree-k+1)
            x = self._rng.beta(k + 1, self.degree - k + 1)
    
            # Compute acceptance probability g^+(x) / h(x)
            gx = np.asarray(self.f(a, x), dtype=float)
            g_plus = np.clip(gx, 0.0, np.inf)
            hx = self._bernstein_eval(b_plus, x)
    
            accept_prob = np.divide(
                g_plus,
                hx,
                out=np.zeros_like(g_plus),
                where=(hx > 0.0),
            )
            accept_prob = np.clip(accept_prob, 0.0, 1.0)
    
            keep = x[self._rng.random(m_batch) <= accept_prob]
    
            n_take = min(size - filled, keep.size)
            if n_take > 0:
                out[filled:filled + n_take] = keep[:n_take]
                filled += n_take
    
        return float(out[0]) if size == 1 else out