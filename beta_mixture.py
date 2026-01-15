import numpy as np
from scipy.stats import beta as beta_dist


class beta_mixture:
    """
    3-component mixture of product-Beta distributions in R^dim.

    Component k has independent coordinates:
        X_j | (Z=k) ~ Beta(alpha[k, j], beta[k, j])

    alpha, beta must be arrays of shape (3, dim).
    probs must be length-3 and sum to 1.
    """

    def __init__(self, dim, alpha, beta, probs, rng=None):
        self.dim = int(dim)

        self.alpha = np.asarray(alpha, dtype=float)
        self.beta = np.asarray(beta, dtype=float)

        if self.alpha.shape != (3, self.dim) or self.beta.shape != (3, self.dim):
            raise ValueError(
                f"alpha and beta must have shape (3, dim) = (3, {self.dim}). "
                f"Got alpha {self.alpha.shape}, beta {self.beta.shape}."
            )

        if np.any(self.alpha <= 0) or np.any(self.beta <= 0):
            raise ValueError("All alpha and beta parameters must be > 0.")

        probs = np.asarray(probs, dtype=float).reshape(-1)
        if probs.shape != (3,):
            raise ValueError(f"probs must have shape (3,), got {probs.shape}.")
        if np.any(probs < 0):
            raise ValueError("All probs must be >= 0.")
        s = probs.sum()
        if not np.isfinite(s) or s <= 0:
            raise ValueError("probs must sum to a positive finite value.")
        probs = probs / s  # normalize just in case
        if not np.isclose(probs.sum(), 1.0, atol=1e-12):
            raise ValueError("probs must sum to 1 (after normalization check failed).")

        self.probs = probs
        self.rng = np.random.default_rng() if rng is None else rng

    def value(self, x_input):
        """
        Mixture PDF at x_input.
        x_input: shape (dim,) or (N, dim), values should be in (0,1).
        Returns: scalar (if input is (dim,)) or shape (N,)
        """
        x = np.asarray(x_input, dtype=float)

        # Ensure shape (N, dim)
        if x.ndim == 1:
            if x.shape[0] != self.dim:
                raise ValueError(f"x_input must have length {self.dim}.")
            x2d = x[None, :]
            squeeze = True
        elif x.ndim == 2:
            if x.shape[1] != self.dim:
                raise ValueError(f"x_input must have shape (N, {self.dim}).")
            x2d = x
            squeeze = False
        else:
            raise ValueError("x_input must be 1D (dim,) or 2D (N, dim).")

        # log p_k(x) = sum_j log BetaPDF(x_j; alpha[k,j], beta[k,j])
        logp0 = beta_dist.logpdf(x2d, self.alpha[0], self.beta[0]).sum(axis=1)
        logp1 = beta_dist.logpdf(x2d, self.alpha[1], self.beta[1]).sum(axis=1)
        logp2 = beta_dist.logpdf(x2d, self.alpha[2], self.beta[2]).sum(axis=1)

        # log mixture: log( sum_k w_k * exp(logp_k) ) via stable max trick
        # allow w_k=0 -> logw=-inf
        logw = np.full(3, -np.inf, dtype=float)
        mask = self.probs > 0
        logw[mask] = np.log(self.probs[mask])

        a = logw[0] + logp0
        b = logw[1] + logp1
        c = logw[2] + logp2

        m = np.maximum(np.maximum(a, b), c)
        out = np.exp(m) * (np.exp(a - m) + np.exp(b - m) + np.exp(c - m))

        return out[0] if squeeze else out

    def generate(self, N):
        """
        Draw N samples, return array of shape (N, dim).
        """
        N = int(N)
        if N <= 0:
            raise ValueError("N must be positive.")

        # Component labels in {0,1,2}
        z = self.rng.choice(3, size=N, p=self.probs)

        X = np.empty((N, self.dim), dtype=float)

        for k in range(3):
            idx = (z == k)
            nk = int(idx.sum())
            if nk > 0:
                X[idx, :] = self.rng.beta(self.alpha[k], self.beta[k], size=(nk, self.dim))

        return X
    
#######
"""
dim=20
alpha1 =  [2*np.ones(dim),  3*np.ones(dim),2*np.ones(dim), ] 

beta1 =  [2 *np.ones(dim),  3*np.ones(dim),2*np.ones(dim), ] 
generator=beta_mixture(dim,   alpha1, beta1, [0.3, 0.3  , 0.4 ])
X_test=generator.generate(10)
generator.value(X_test)
""" 