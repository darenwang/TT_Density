import numpy as np


def gram_matrix_one_hat(n):
    """
    Gram matrix on [0,1] for the collection
        {1, phi_0, phi_1, ..., phi_{n-2}}
    where phi_i are hat functions on the uniform mesh
        x_i = i / (n - 1),   i = 0, 1, ..., n - 1.

    Thus the total number of basis functions is exactly n:
        1 constant + (n - 1) hat functions.

    Parameters
    ----------
    n : int
        Total number of basis functions.

    Returns
    -------
    G : ndarray of shape (n, n)
        Gram matrix with basis ordering
        [1, phi_0, phi_1, ..., phi_{n-2}].
    """
    if n < 1:
        raise ValueError("n must be a positive integer.")

    # Special case: only the constant basis
    if n == 1:
        return np.array([[1.0]], dtype=float)

    h = 1.0 / (n - 1)
    G = np.zeros((n, n), dtype=float)

    # <1, 1>
    G[0, 0] = 1.0

    # <1, phi_i>
    # phi_0 is the left boundary hat, integral = h/2
    G[0, 1] = h / 2.0
    G[1, 0] = h / 2.0

    # phi_i for i = 1, ..., n-2 each has integral h
    for i in range(1, n - 1):
        G[0, i + 1] = h
        G[i + 1, 0] = h

    # Hat-hat block
    # diagonal
    G[1, 1] = h / 3.0              # <phi_0, phi_0>
    for i in range(2, n):
        G[i, i] = 2.0 * h / 3.0    # <phi_i, phi_i>, i=1,...,n-2

    # nearest-neighbor off-diagonal
    for i in range(1, n - 1):
        G[i, i + 1] = h / 6.0
        G[i + 1, i] = h / 6.0

    return G


################################
class polynomial:
    def __init__(self, n, alpha) -> None:
        """
        Basis collection:
            {1, phi_0, phi_1, ..., phi_{n-2}}

        So the total number of basis functions is exactly n.
        """
        if n < 1:
            raise ValueError("n must be a positive integer.")

        self.n = n
        self.num_basis = n

        G = gram_matrix_one_hat(n)
        self.L = np.linalg.cholesky(G)
        self.Linvt = np.linalg.inv(self.L).T

        alpha_vec = alpha * np.ones(self.num_basis, dtype=float)
        alpha_vec[0] = 1.0
        self.alpha_vec = alpha_vec

    def _hat_basis_values(self, x):
        """
        Evaluate the basis {1, phi_0, ..., phi_{n-2}}.

        Parameters
        ----------
        x : array_like
            1D array of points in [0,1].

        Returns
        -------
        basis : ndarray, shape (len(x), n)
            basis[:, 0]   = 1
            basis[:, i+1] = phi_i,  i=0,...,n-2
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        n = self.n

        basis = np.zeros((x.size, n), dtype=float)
        basis[:, 0] = ((x >= 0.0) & (x <= 1.0)).astype(float)

        # If n == 1, there are no hat functions
        if n == 1:
            return basis

        m = n - 1          # number of subintervals
        h = 1.0 / m

        # phi_0
        mask0 = (x >= 0.0) & (x <= h)
        basis[mask0, 1] = 1.0 - x[mask0] / h

        # phi_i for i = 1, ..., n-2
        for i in range(1, n - 1):
            left = (i - 1) * h
            mid = i * h
            right = (i + 1) * h

            mask_left = (x >= left) & (x <= mid)
            basis[mask_left, i + 1] = (x[mask_left] - left) / h

            mask_right = (x >= mid) & (x <= right)
            right_vals = (right - x[mask_right]) / h
            basis[mask_right, i + 1] = np.maximum(
                basis[mask_right, i + 1],
                right_vals
            )

        return basis

    def multivariate_all_basis_alpha(self, x):
        basis = self._hat_basis_values(x)
        return (basis @ self.Linvt) * self.alpha_vec[None, :]

    def scalar_all_basis_alpha(self, x):
        return self.multivariate_all_basis_alpha(np.array([x], dtype=float))[0]


##########


class generate_basis_mat:
    def __init__(self, n, dim, alpha, data):
        self.n = n
        self.polynomial = polynomial(n, alpha)
        self.dim = dim
        self.data = data

    def compute(self):
        """
        Returns
        -------
        basis_mat : ndarray of shape (len(data), dim, n)
            At each scalar input, returns the n basis values
            corresponding to {1, phi_0, ..., phi_{n-2}}.
        """
        new_data = self.data.reshape(-1)
        basis_mat_flat = self.polynomial.multivariate_all_basis_alpha(new_data)
        return basis_mat_flat.reshape(len(self.data), self.dim, self.n)
