import numpy as np 

from scipy import stats


 
    

class kernel_density:
    def __init__(self, data):
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError("data must be a 2D array of shape (n_samples, dim)")
        
        self.data = data
        self.n, self.dim = data.shape
        self.kernel = stats.gaussian_kde(data.T)

    def compute(self, X_new):
        X_new = np.asarray(X_new, dtype=float)
        return self.kernel(X_new.T)

    def conditional_sample(self, x_given, N, seed=None):
        """
        Sample from the conditional KDE distribution given that the first
        k = len(x_given) coordinates are fixed at x_given.

        Parameters
        ----------
        x_given : array-like, shape (k,)
            Values of the first k coordinates to condition on.
        N : int
            Number of conditional samples.
        seed : int or None
            Random seed.

        Returns
        -------
        samples : ndarray, shape (N, dim)
            Full conditional samples. The first k columns are exactly x_given.
        """
        rng = np.random.default_rng(seed)

        x_given = np.asarray(x_given, dtype=float).reshape(-1)
        k = len(x_given)

        if not (0 < k < self.dim):
            raise ValueError("Need 0 < len(x_given) < dim.")

        # KDE representation:
        # p(x) = sum_i w_i N(x | data_i, H)
        # where H is the common kernel covariance matrix
        X = self.kernel.dataset.T          # shape (n, dim)
        H = self.kernel.covariance         # shape (dim, dim)

        if hasattr(self.kernel, "weights") and self.kernel.weights is not None:
            base_weights = np.asarray(self.kernel.weights, dtype=float)
        else:
            base_weights = np.ones(self.n, dtype=float) / self.n

        # Partition coordinates into given part and remaining part
        H11 = H[:k, :k]
        H12 = H[:k, k:]
        H21 = H[k:, :k]
        H22 = H[k:, k:]

        X1 = X[:, :k]     # shape (n, k)
        X2 = X[:, k:]     # shape (n, dim-k)

        # For each mixture component i, compute weight proportional to
        # w_i * N(x_given | X1[i], H11)
        inv_H11 = np.linalg.inv(H11)
        sign, logdet_H11 = np.linalg.slogdet(H11)
        if sign <= 0:
            raise ValueError("H11 is not positive definite.")

        diff = x_given[None, :] - X1   # shape (n, k)
        quad = np.sum((diff @ inv_H11) * diff, axis=1)

        log_norm_const = -0.5 * (k * np.log(2 * np.pi) + logdet_H11)
        log_component_density = log_norm_const - 0.5 * quad

        log_weights = np.log(base_weights) + log_component_density
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= np.sum(weights)

        # Conditional covariance is the same for every component
        # H_{2|1} = H22 - H21 H11^{-1} H12
        H_cond = H22 - H21 @ inv_H11 @ H12
        H_cond = 0.5 * (H_cond + H_cond.T)  # numerical symmetrization

        # Conditional mean for each component:
        # mu_i = X2[i] + H21 H11^{-1} (x_given - X1[i])
        A = H21 @ inv_H11                   # shape (dim-k, k)
        cond_means = X2 + (diff @ A.T)      # shape (n, dim-k)

        # Sample mixture component indices
        comp_idx = rng.choice(self.n, size=N, p=weights)

        # Sample remaining coordinates
        L = np.linalg.cholesky(H_cond)
        Z = rng.standard_normal((N, self.dim - k))
        sampled_rest = cond_means[comp_idx] + Z @ L.T

        # Return full samples
        x_fixed = np.tile(x_given, (N, 1))
        samples = np.hstack([x_fixed, sampled_rest])
        return samples