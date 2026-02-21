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



###############
###############
############### conditional sampling



from scipy.special import betaln, logsumexp




def conditional_sample_beta_mixture(alpha, beta, weights, x_given, N_sample,
                                    observed_idx=None, clip_eps=1e-12, rng=None,
                                    return_posterior=False, check_output=False):
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    if alpha.ndim != 2 or beta.ndim != 2:
        raise ValueError("alpha and beta must be 2D arrays of shape (K, dim).")
    if alpha.shape != beta.shape:
        raise ValueError("alpha and beta must have the same shape (K, dim).")
    K, dim = alpha.shape

    if weights.shape != (K,):
        raise ValueError(f"weights must have shape ({K},).")
    if np.any(alpha <= 0) or np.any(beta <= 0):
        raise ValueError("All alpha and beta entries must be > 0.")
    if np.any(weights < 0):
        raise ValueError("weights must be nonnegative.")
    wsum = weights.sum()
    if not np.isfinite(wsum) or wsum <= 0:
        raise ValueError("weights must sum to a positive finite value.")
    weights = weights / wsum

    x_given = np.asarray(x_given, dtype=float).reshape(-1)
    m = x_given.size
    if m == 0:
        raise ValueError("x_given must be nonempty.")
    if observed_idx is None:
        observed_idx = np.arange(m, dtype=int)
    else:
        observed_idx = np.asarray(observed_idx, dtype=int).reshape(-1)

    if observed_idx.size != m:
        raise ValueError("observed_idx and x_given must have the same length.")
    if np.any(observed_idx < 0) or np.any(observed_idx >= dim):
        raise ValueError("observed_idx contains out-of-range indices.")
    if len(np.unique(observed_idx)) != observed_idx.size:
        raise ValueError("observed_idx contains duplicates.")

    # Clip ONLY for likelihood evaluation stability
    eps = float(clip_eps)
    x_obs = np.clip(x_given, eps, 1.0 - eps)

    a_obs = alpha[:, observed_idx]   # (K, m)
    b_obs = beta[:, observed_idx]    # (K, m)

    log_pdf = (a_obs - 1.0) * np.log(x_obs)[None, :] + (b_obs - 1.0) * np.log1p(-x_obs)[None, :] - betaln(a_obs, b_obs)
    log_like = log_pdf.sum(axis=1)  # (K,)

    logw = np.full(K, -np.inf, dtype=float)
    mask = weights > 0
    logw[mask] = np.log(weights[mask])

    log_post_unnorm = logw + log_like
    log_post = log_post_unnorm - logsumexp(log_post_unnorm)
    post = np.exp(log_post)

    if rng is None:
        rng = np.random.default_rng()

    N_sample = int(N_sample)
    if N_sample <= 0:
        raise ValueError("N_sample must be positive.")

    z = rng.choice(K, size=N_sample, p=post)

    X = np.empty((N_sample, dim), dtype=float)
    X[:, observed_idx] = x_given[None, :]  # note: if x_given is outside [0,1], these entries will be outside too

    unobs_idx = np.setdiff1d(np.arange(dim, dtype=int), observed_idx)
    if unobs_idx.size > 0:
        for k in range(K):
            rows = np.where(z == k)[0]
            if rows.size == 0:
                continue
            A = alpha[k, unobs_idx]
            B = beta[k, unobs_idx]
            draws = rng.beta(A, B, size=(rows.size, unobs_idx.size))
            X[np.ix_(rows, unobs_idx)] = draws  # <-- correct in-place assignment

    if check_output:
        # This checks only the *unobserved* coordinates are in [0,1]
        if unobs_idx.size > 0 and (np.any(X[:, unobs_idx] < 0) or np.any(X[:, unobs_idx] > 1)):
            raise RuntimeError("Bug: unobserved Beta draws fell outside [0,1].")

    if return_posterior:
        return X, post
    return X
