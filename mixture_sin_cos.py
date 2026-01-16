import numpy as np


class SinPlusOneBox:
    """
    dim-dimensional distribution on [-1,1]^dim with independent coordinates.

    1D normalized density:
        f1(x) = (1 + sin(alpha*x)) / 2   on [-1,1]
    Joint:
        f(x) = Π_j f1(x_j)
    """
    def __init__(self, dim: int, alpha: float):
        self.dim = int(dim)
        self.alpha = float(alpha)
        if self.dim <= 0:
            raise ValueError("dim must be a positive integer.")

    def _pdf1(self, x):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        mask = (x >= -1.0) & (x <= 1.0)
        if np.isclose(self.alpha, 0.0):
            out[mask] = 0.5
        else:
            out[mask] = 0.5 * (1.0 + np.sin(self.alpha * x[mask]))
        return out

    def _sample_1d(self, n: int):
        n = int(n)
        if n < 0:
            raise ValueError("n must be nonnegative.")
        if n == 0:
            return np.empty((0,), dtype=float)
        if np.isclose(self.alpha, 0.0):
            return np.random.uniform(-1.0, 1.0, size=n)

        samples = np.empty(n, dtype=float)
        filled = 0
        while filled < n:
            need = n - filled
            m = int(np.ceil(2.2 * need)) + 10  # accept rate ~ 1/2
            x = np.random.uniform(-1.0, 1.0, size=m)
            u = np.random.uniform(0.0, 1.0, size=m)
            acc = u <= 0.5 * (1.0 + np.sin(self.alpha * x))
            accepted = x[acc]
            k = min(need, accepted.size)
            if k > 0:
                samples[filled:filled + k] = accepted[:k]
                filled += k
        return samples

    def generate(self, N: int):
        N = int(N)
        if N < 0:
            raise ValueError("N must be nonnegative.")
        x = self._sample_1d(N * self.dim)
        return x.reshape(N, self.dim)

    def value(self, x):
        x = np.asarray(x, dtype=float)
        scalar_input = False
        if x.ndim == 1:
            if x.shape[0] != self.dim:
                raise ValueError(f"x must have length dim={self.dim}.")
            x = x[None, :]
            scalar_input = True
        elif x.ndim == 2:
            if x.shape[1] != self.dim:
                raise ValueError(f"x must have shape (M, {self.dim}).")
        else:
            raise ValueError("x must be a 1D or 2D array.")

        dens = self._pdf1(x)          # (M, dim)
        out = np.prod(dens, axis=1)   # (M,)
        return out[0] if scalar_input else out


class CosPlusOneBox:
    """
    dim-dimensional distribution on [-1,1]^dim with independent coordinates.

    Unnormalized 1D density: 1 + cos(beta*x) on [-1,1].

    Normalizer:
        Z = ∫_{-1}^1 (1 + cos(beta x)) dx
          = 2 + 2*sin(beta)/beta   (beta != 0)
          = 4                      (beta = 0, since 1+cos(0)=2)

    1D normalized density:
        f1(x) = (1 + cos(beta*x))/Z on [-1,1]
        f1(x) = 1/2                 if beta = 0

    Joint:
        f(x) = Π_j f1(x_j)
    """
    def __init__(self, dim: int, beta: float):
        self.dim = int(dim)
        self.beta = float(beta)
        if self.dim <= 0:
            raise ValueError("dim must be a positive integer.")
        self._Z = self._norm_const_1d()

    def _norm_const_1d(self) -> float:
        if np.isclose(self.beta, 0.0):
            return 4.0
        return 2.0 + 2.0 * np.sin(self.beta) / self.beta

    def _pdf1(self, x):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        mask = (x >= -1.0) & (x <= 1.0)
        if np.isclose(self.beta, 0.0):
            out[mask] = 0.5
        else:
            out[mask] = (1.0 + np.cos(self.beta * x[mask])) / self._Z
        return out

    def _sample_1d(self, n: int):
        n = int(n)
        if n < 0:
            raise ValueError("n must be nonnegative.")
        if n == 0:
            return np.empty((0,), dtype=float)
        if np.isclose(self.beta, 0.0):
            return np.random.uniform(-1.0, 1.0, size=n)

        # Rejection from Uniform(-1,1)
        # max f = 2/Z, g=1/2 => M = (2/Z)/(1/2)=4/Z
        M = 4.0 / self._Z
        samples = np.empty(n, dtype=float)
        filled = 0
        while filled < n:
            need = n - filled
            m = int(np.ceil(M * need * 1.2)) + 10
            x = np.random.uniform(-1.0, 1.0, size=m)
            u = np.random.uniform(0.0, 1.0, size=m)
            fx = (1.0 + np.cos(self.beta * x)) / self._Z
            acc = u <= (2.0 * fx / M)  # fx / (M*g) with g=1/2
            accepted = x[acc]
            k = min(need, accepted.size)
            if k > 0:
                samples[filled:filled + k] = accepted[:k]
                filled += k
        return samples

    def generate(self, N: int):
        N = int(N)
        if N < 0:
            raise ValueError("N must be nonnegative.")
        x = self._sample_1d(N * self.dim)
        return x.reshape(N, self.dim)

    def value(self, x):
        x = np.asarray(x, dtype=float)
        scalar_input = False
        if x.ndim == 1:
            if x.shape[0] != self.dim:
                raise ValueError(f"x must have length dim={self.dim}.")
            x = x[None, :]
            scalar_input = True
        elif x.ndim == 2:
            if x.shape[1] != self.dim:
                raise ValueError(f"x must have shape (M, {self.dim}).")
        else:
            raise ValueError("x must be a 1D or 2D array.")

        dens = self._pdf1(x)
        out = np.prod(dens, axis=1)
        return out[0] if scalar_input else out


class sin_cos_mixture:
    """
    Mixture of SinPlusOneBox and CosPlusOneBox on [-1,1]^dim.

    weights: array-like of shape (2,)
      - weights[0] for SinPlusOneBox
      - weights[1] for CosPlusOneBox
    Mixture density:
        p(x) = w0 * p_sin(x) + w1 * p_cos(x)
    """

    def __init__(self, dim: int, alpha: float, beta: float, weights):
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError("dim must be a positive integer.")

        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size != 2:
            raise ValueError("weights must be a vector of size 2.")
        if np.any(w < 0):
            raise ValueError("weights must be nonnegative.")
        s = w.sum()
        if s <= 0:
            raise ValueError("weights must sum to a positive number.")
        self.weights = w / s

        self.sin_comp = SinPlusOneBox(dim=self.dim, alpha=alpha)
        self.cos_comp = CosPlusOneBox(dim=self.dim, beta=beta)

    def generate(self, N: int):
        """
        Generate N samples from the mixture.
        Output shape: (N, dim)
        """
        N = int(N)
        if N < 0:
            raise ValueError("N must be nonnegative.")
        if N == 0:
            return np.empty((0, self.dim), dtype=float)

        # component labels
        z = np.random.choice(2, size=N, p=self.weights)
        n0 = int(np.sum(z == 0))
        n1 = N - n0

        X = np.empty((N, self.dim), dtype=float)
        if n0 > 0:
            X[z == 0] = self.sin_comp.generate(n0)
        if n1 > 0:
            X[z == 1] = self.cos_comp.generate(n1)
        return X

    def value(self, x):
        """
        Evaluate mixture density at x.
        x: (M, dim) or (dim,)
        returns: (M,) density values (or scalar if input is (dim,))
        """
        x = np.asarray(x, dtype=float)
        scalar_input = (x.ndim == 1)
        ps = self.sin_comp.value(x)  # (M,) or scalar
        pc = self.cos_comp.value(x)
        out = self.weights[0] * ps + self.weights[1] * pc
        return out if not scalar_input else float(out)
