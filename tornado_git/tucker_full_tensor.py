import numpy as np
from scipy.linalg import svd

from local_basis import generate_basis_mat, polynomial
from sampling_local import Legendre_Sampler


def estimate_tensor_memory_gb(n, dim, dtype=np.float64):
    """
    Estimate memory needed to store a full tensor of shape (n, ..., n).

    Parameters
    ----------
    n : int
        Number of basis functions per coordinate.
    dim : int
        Dimension of the tensor.
    dtype : data-type
        Data type used to store the tensor.

    Returns
    -------
    memory_gb : float
        Estimated memory in GB.
    """
    return (n ** dim) * np.dtype(dtype).itemsize / (1024 ** 3)


def build_full_tensor_from_basis(basis_mat):
    """
    Build the full coefficient tensor

        A[i1, ..., id]
        =
        (1/N) sum_{ell=1}^N
        B_1[ell, i1] ... B_d[ell, id].

    Parameters
    ----------
    basis_mat : list or ndarray
        basis_mat[j] has shape (N, n). If an ndarray is passed, it should
        have shape (dim, N, n).

    Returns
    -------
    tensor : ndarray, shape (n, ..., n)
        Full coefficient tensor.
    """
    basis_mat = [np.asarray(B, dtype=float) for B in basis_mat]

    dim = len(basis_mat)
    if dim == 0:
        raise ValueError("basis_mat must contain at least one mode.")

    N, n = basis_mat[0].shape

    for j in range(dim):
        if basis_mat[j].ndim != 2:
            raise ValueError("Each basis matrix must be 2D.")
        if basis_mat[j].shape != (N, n):
            raise ValueError("All basis matrices must have the same shape (N, n).")

    # z is the sample index.
    # Other labels are tensor-mode indices.
    mode_labels = "abcdefghijklmnopqrstuvwxyABCDEFGHIJKLMNOPQRSTUVWXY"

    if dim > len(mode_labels):
        raise ValueError("dim is too large for this einsum construction.")

    input_terms = ["z" + mode_labels[j] for j in range(dim)]
    output_term = "".join(mode_labels[j] for j in range(dim))

    einsum_str = ",".join(input_terms) + "->" + output_term

    tensor = np.einsum(einsum_str, *basis_mat, optimize=True) / N

    return tensor


def mode_unfold(tensor, mode):
    """
    Compute the mode-k unfolding.

    Parameters
    ----------
    tensor : ndarray
        Input tensor.
    mode : int
        Mode to unfold.

    Returns
    -------
    unfolding : ndarray
        Matrix of shape

            tensor.shape[mode] by product of all other mode sizes.
    """
    tensor = np.asarray(tensor)
    tensor_mode_first = np.moveaxis(tensor, mode, 0)
    return tensor_mode_first.reshape(tensor.shape[mode], -1)


def mode_dot(tensor, matrix, mode):
    """
    Multiply a tensor by a matrix along one mode.

    Parameters
    ----------
    tensor : ndarray
        Input tensor.
    matrix : ndarray
        Matrix of shape (new_dim, old_dim).
    mode : int
        Mode of tensor to multiply.

    Returns
    -------
    result : ndarray
        Tensor after mode multiplication.
    """
    tensor = np.asarray(tensor)
    matrix = np.asarray(matrix)

    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D.")
    if matrix.shape[1] != tensor.shape[mode]:
        raise ValueError(
            "matrix.shape[1] must equal tensor.shape[mode]. "
            f"Got {matrix.shape[1]} and {tensor.shape[mode]}."
        )

    result = np.tensordot(matrix, tensor, axes=([1], [mode]))

    # tensordot places the new mode in the first axis.
    # Move it back to the original mode location.
    result = np.moveaxis(result, 0, mode)

    return result


def choose_rank_from_singular_values(
    singular_values,
    threshold=0.9999,
    rank_rule="energy",
    min_rank=1,
):
    """
    Choose Tucker rank from singular values.

    Parameters
    ----------
    singular_values : ndarray
        Singular values from a mode unfolding.
    threshold : float
        Threshold value.
    rank_rule : str
        Rank selection rule.

        "energy":
            Keep enough components so that cumulative squared singular-value
            energy is at least threshold. For example, threshold=0.9999.

        "relative":
            Keep singular values S_j satisfying S_j >= threshold * S_1.
            For example, threshold=1e-6.

        "absolute":
            Keep singular values S_j satisfying S_j >= threshold.

        "eigen_absolute":
            Keep components satisfying S_j^2 >= threshold.
            This matches thresholding eigenvalues of unfolding @ unfolding.T.

        "fixed":
            Interpret threshold as the fixed rank.

    min_rank : int
        Minimum rank to keep.

    Returns
    -------
    rank : int
        Selected rank.
    """
    s = np.asarray(singular_values, dtype=float)

    if s.size == 0:
        return 0

    if rank_rule == "energy":
        energy = s ** 2
        total_energy = np.sum(energy)

        if total_energy <= 0:
            rank = min_rank
        else:
            cumulative_energy = np.cumsum(energy) / total_energy
            rank = np.searchsorted(cumulative_energy, threshold) + 1

    elif rank_rule == "relative":
        if s[0] <= 0:
            rank = min_rank
        else:
            rank = np.sum(s >= threshold * s[0])

    elif rank_rule == "absolute":
        rank = np.sum(s >= threshold)

    elif rank_rule == "eigen_absolute":
        rank = np.sum(s ** 2 >= threshold)

    elif rank_rule == "fixed":
        rank = int(threshold)

    else:
        raise ValueError(
            "Unknown rank_rule. Use one of "
            "'energy', 'relative', 'absolute', 'eigen_absolute', or 'fixed'."
        )

    rank = int(rank)
    rank = max(rank, min_rank)
    rank = min(rank, len(s))

    return rank


def tucker_hosvd_from_full_tensor(
    tensor,
    threshold=0.9999,
    rank_rule="energy",
    min_rank=1,
    verbose=True,
):
    """
    Compute Tucker decomposition by HOSVD from the full tensor.

    Parameters
    ----------
    tensor : ndarray
        Full tensor of shape (n, ..., n).
    threshold : float
        Threshold used for rank selection.
    rank_rule : str
        Rank selection rule. See choose_rank_from_singular_values.
    min_rank : int
        Minimum rank in each mode.
    verbose : bool
        Whether to print singular values and ranks.

    Returns
    -------
    core : ndarray
        Tucker core of shape (r_1, ..., r_d).
    mat_list : list of ndarray
        mat_list[k] has shape (n, r_k).
    ranks : list of int
        Tucker ranks.
    singular_values_list : list of ndarray
        Singular values from each mode unfolding.
    """
    tensor = np.asarray(tensor, dtype=float)
    dim = tensor.ndim

    mat_list = []
    ranks = []
    singular_values_list = []

    for mode in range(dim):
        unfolding = mode_unfold(tensor, mode)

        U, S, Vt = svd(
            unfolding,
            full_matrices=False,
            check_finite=False,
        )

        rank_k = choose_rank_from_singular_values(
            S,
            threshold=threshold,
            rank_rule=rank_rule,
            min_rank=min_rank,
        )

        G_cur = U[:, :rank_k]

        mat_list.append(G_cur)
        ranks.append(rank_k)
        singular_values_list.append(S)

        if verbose:
            print(f"mode {mode}: singular values = {S}")
            print(f"mode {mode}: rank = {rank_k}")

    # Project full tensor onto the Tucker factors:
    #
    #     core = A x_1 U_1^T x_2 U_2^T ... x_d U_d^T.
    core = tensor.copy()
    for mode in range(dim):
        core = mode_dot(core, mat_list[mode].T, mode)

    return core, mat_list, ranks, singular_values_list


class Tucker:
    """
    Tucker density estimator using the explicit full tensor.

    This version first builds the full tensor of size n^dim and then computes
    a Tucker decomposition by HOSVD. This is simple and useful for debugging,
    but it is only feasible when n^dim is moderate.

    The full tensor is

        A[i1, ..., id]
        =
        (1/N) sum_{ell=1}^N
        B_1[ell, i1] ... B_d[ell, id].

    Parameters
    ----------
    n : int
        Number of basis functions per coordinate.
    X_train : ndarray, shape (N, dim)
        Training data.
    max_iterate : int
        Kept only for compatibility with your old interface. This explicit
        HOSVD version does not use ALS iterations.
    threshold : float
        Threshold used for rank selection.
    rank_rule : str
        One of "energy", "relative", "absolute", "eigen_absolute", or "fixed".
    max_memory_gb : float
        Safety limit for full tensor memory.
    min_rank : int
        Minimum Tucker rank in each mode.
    verbose : bool
        Whether to print diagnostic information.
    """

    def __init__(
        self,
        n,
        X_train,
        max_iterate=0,
        threshold=0.99999,
        rank_rule="energy",
        max_memory_gb=8.0,
        min_rank=1,
        verbose=True,
    ):
        self.n = int(n)
        self.threshold = threshold
        self.rank_rule = rank_rule
        self.min_rank = int(min_rank)
        self.verbose = bool(verbose)

        X_train = np.asarray(X_train, dtype=float)
        if X_train.ndim != 2:
            raise ValueError("X_train must have shape (N, dim).")

        self.N, self.dim = X_train.shape

        self.new_domain = domain(self.dim, X_train)
        self.X_train_transform = self.new_domain.transform_data(X_train)
        self.X_train_transform = np.clip(self.X_train_transform, 0.0, 1.0)

        # generate_basis_mat(...).compute() is assumed to have shape (N, dim, n).
        # After transpose, self.basis_mat has shape (dim, N, n).
        self.basis_mat = generate_basis_mat(
            self.n, self.dim, 1, self.X_train_transform
        ).compute().transpose(1, 0, 2)

        memory_gb = estimate_tensor_memory_gb(self.n, self.dim)

        if self.verbose:
            print(f"Full tensor shape: ({self.n},)^{self.dim}")
            print(f"Approximate full tensor memory: {memory_gb:.6f} GB")

        if memory_gb > max_memory_gb:
            raise MemoryError(
                f"The full tensor needs about {memory_gb:.2f} GB, "
                f"which exceeds max_memory_gb={max_memory_gb}. "
                "Reduce n or dim, increase max_memory_gb, or use the implicit "
                "Gram-matrix version instead."
            )

        # Step 1: build the full tensor of size n^dim.
        self.full_tensor = build_full_tensor_from_basis(self.basis_mat)

        # Step 2: compute Tucker decomposition by HOSVD.
        self.core, self.mat_list, self.ranks, self.singular_values_list = (
            tucker_hosvd_from_full_tensor(
                self.full_tensor,
                threshold=self.threshold,
                rank_rule=self.rank_rule,
                min_rank=self.min_rank,
                verbose=self.verbose,
            )
        )

        if self.verbose:
            print("Tucker ranks:", self.ranks)

        self.polynomial = polynomial(self.n, 1)
        self.sampler = Legendre_Sampler(self.n, self.polynomial.Linvt)

    def predict(self, X_test):
        """
        Evaluate the fitted density at test points.

        Parameters
        ----------
        X_test : ndarray, shape (N_test, dim)

        Returns
        -------
        y_pred : ndarray, shape (N_test,)
        """
        X_test = np.asarray(X_test, dtype=float)
        if X_test.ndim != 2:
            raise ValueError("X_test must have shape (N_test, dim).")
        if X_test.shape[1] != self.dim:
            raise ValueError("X_test.shape[1] must equal dim.")

        X_test_transform = self.new_domain.transform_data(X_test)
        X_test_transform = np.clip(X_test_transform, 0.0, 1.0)

        mat_test = generate_basis_mat(
            self.n, self.dim, 1, X_test_transform
        ).compute()

        y_pred = Tucker_prediction().predict(
            self.dim, self.core, self.mat_list, mat_test
        )

        y_pred = self.new_domain.transform_density_val(y_pred)

        return np.clip(y_pred, 1e-14, np.inf)

    def sample_one(self):
        """
        Draw one sample from the fitted Tucker density on the transformed
        [0, 1]^dim scale, then return it to the original data scale in sample().
        """
        result = np.zeros(self.dim)
        condition_mat = self.core

        for d in range(self.dim - 1):
            tail_vec = self.compute_marginal(condition_mat, first_mode=d)
            cur_vec = self.mat_list[d] @ tail_vec

            result[d] = self.sampler.sample(cur_vec)
            if result[d] == -np.inf:
                return False, []

            cur_basis_val = self.polynomial.scalar_all_basis_alpha(result[d])
            cur_basis_val_r = cur_basis_val @ self.mat_list[d]

            condition_mat = np.tensordot(
                condition_mat,
                cur_basis_val_r,
                axes=([0], [0]),
            )

        cur_vec = self.mat_list[-1] @ condition_mat
        result[-1] = self.sampler.sample(cur_vec)
        if result[-1] == -np.inf:
            return False, []

        return True, result

    def sample(self, N_sample, verbose=True):
        """
        Draw samples from the fitted Tucker density.

        Parameters
        ----------
        N_sample : int
            Number of samples.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        result : ndarray, shape (N_sample, dim)
            Samples on the original data scale.
        """
        N_sample = int(N_sample)
        result = np.zeros((N_sample, self.dim))

        for i in range(N_sample):
            if verbose and i % 1000 == 0:
                print(i)

            ok, sample_point = self.sample_one()
            while not ok:
                ok, sample_point = self.sample_one()

            result[i] = sample_point

        return self.new_domain.inverse_compute_data(result)

    def compute_marginal(self, cur_tensor, first_mode):
        """
        Compute marginal vector needed for sequential conditional sampling.

        Parameters
        ----------
        cur_tensor : ndarray
            Current conditional tensor. Its axes correspond to original modes

                first_mode, first_mode + 1, ..., dim - 1.

        first_mode : int
            First original mode represented by cur_tensor.

        Returns
        -------
        temp_tensor : ndarray
            Marginal vector/tensor after integrating out later coordinates.
        """
        temp_tensor = cur_tensor

        for mode in range(self.dim - 1, first_mode, -1):
            temp_tensor = np.tensordot(
                temp_tensor,
                self.mat_list[mode][0, :],
                axes=([-1], [0]),
            )

        return temp_tensor


class domain:
    """
    Affine transformation between original data domain and [0, 1]^dim.

    The fitted density is learned on [0, 1]^dim. Density values are then
    rescaled by the Jacobian factor.
    """

    def __init__(self, dim, X_train):
        factor = 10 ** (-5)

        self.X_train = np.asarray(X_train, dtype=float)
        self.dim = int(dim)

        self.upper = []
        self.lower = []

        for dd in range(self.dim):
            self.upper.append(np.quantile(self.X_train[:, dd], 1 - factor))
            self.lower.append(np.quantile(self.X_train[:, dd], factor))

        self.upper = np.array(self.upper)
        self.lower = np.array(self.lower)

        self.difference = self.upper - self.lower

        # Avoid division by zero in degenerate coordinates.
        self.difference = np.maximum(self.difference, 1e-12)

        self.density_factor = np.prod(self.difference)

    def transform_density_val(self, val):
        return val / self.density_factor

    def transform_data(self, XX):
        XX = np.asarray(XX, dtype=float)
        return (XX - self.lower) / self.difference

    def transform_partial_data(self, begin_dim, end_dim, XX):
        XX = np.asarray(XX, dtype=float)
        return (
            XX - self.lower[begin_dim:end_dim]
        ) / self.difference[begin_dim:end_dim]

    def inverse_compute_data(self, UU):
        UU = np.asarray(UU, dtype=float)
        return UU * self.difference + self.lower

    def inverse_partial_data(self, begin_dim, end_dim, UU):
        UU = np.asarray(UU, dtype=float)
        return UU * self.difference[begin_dim:end_dim] + self.lower[begin_dim:end_dim]


class Tucker_prediction:
    """
    Vectorized Tucker prediction.
    """

    def predict(self, dim, core, mat_list, mat_test):
        """
        Parameters
        ----------
        dim : int
            Number of dimensions.
        core : ndarray
            Tucker core of shape (ranks[0], ..., ranks[dim - 1]).
        mat_list : list of ndarray
            mat_list[j] has shape (n, ranks[j]).
        mat_test : ndarray
            Basis values at test points, shape (N_test, dim, n).

        Returns
        -------
        y_pred : ndarray, shape (N_test,)
            Predicted density values on transformed scale.
        """
        if mat_test.ndim != 3:
            raise ValueError("mat_test must have shape (N_test, dim, n).")
        if mat_test.shape[1] != dim:
            raise ValueError("mat_test.shape[1] must equal dim.")
        if len(mat_list) != dim:
            raise ValueError("mat_list must have length dim.")

        W_list = [mat_test[:, j, :] @ mat_list[j] for j in range(dim)]

        sample_label = "z"
        rank_labels = "abcdefghijklmnopqrstuvwxyABCDEFGHIJKLMNOPQRSTUVWXY"

        if dim > len(rank_labels):
            raise ValueError("dim is too large for this einsum construction.")

        core_term = "".join(rank_labels[j] for j in range(dim))
        input_terms = [core_term]

        for j in range(dim):
            input_terms.append(sample_label + rank_labels[j])

        einsum_str = ",".join(input_terms) + "->" + sample_label

        y_pred = np.einsum(einsum_str, core, *W_list, optimize=True)

        return y_pred


if __name__ == "__main__":
    # Minimal example.
    #
    # This requires your local_basis.py and sampling_local.py files to be
    # available in the same Python environment.
    #
    # Use small n and dim first because the full tensor has size n^dim.
    rng = np.random.default_rng(123)

    N = 200
    dim = 3
    n = 6

    X_train = rng.normal(size=(N, dim))

    model = Tucker(
        n=n,
        X_train=X_train,
        threshold=0.9999,
        rank_rule="energy",
        max_memory_gb=1.0,
        verbose=True,
    )

    X_test = rng.normal(size=(10, dim))
    y_pred = model.predict(X_test)

    print("Prediction shape:", y_pred.shape)
    print("Predictions:", y_pred)
