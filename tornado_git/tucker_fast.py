import numpy as np
from scipy.linalg import eigh

from local_basis import generate_basis_mat, polynomial
from sampling_local import Legendre_Sampler


def _as_basis_list(basis_mat):
    """
    Convert basis_mat to a list of arrays.

    Expected shape:
        basis_mat[j] has shape (N, n), j = 0, ..., dim - 1.

    If an ndarray is passed, it may have shape (dim, N, n).
    """
    if isinstance(basis_mat, np.ndarray):
        if basis_mat.ndim != 3:
            raise ValueError("basis_mat ndarray must have shape (dim, N, n).")
        basis_mat = [basis_mat[j] for j in range(basis_mat.shape[0])]

    basis_mat = [np.asarray(B, dtype=float) for B in basis_mat]

    if len(basis_mat) == 0:
        raise ValueError("basis_mat must contain at least one mode.")

    N, n = basis_mat[0].shape
    for j, B in enumerate(basis_mat):
        if B.ndim != 2:
            raise ValueError(f"basis_mat[{j}] must be 2D.")
        if B.shape != (N, n):
            raise ValueError("All basis matrices must have the same shape (N, n).")

    return basis_mat


def choose_rank_from_singular_values(
    singular_values,
    threshold=0.9999,
    rank_rule="energy",
    min_rank=1,
):
    """
    Choose Tucker rank from singular values.

    rank_rule:
        "energy": keep enough squared singular-value energy.
        "relative": keep S_j >= threshold * S_1.
        "absolute": keep S_j >= threshold.
        "eigen_absolute": keep S_j^2 >= threshold.
        "fixed": use threshold as the fixed rank.
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


def compute_basis_row_grams(basis_mat):
    """
    Compute H_j = B_j B_j^T for each mode.

    Each H_j has shape (N, N). These are much smaller than the full tensor
    when n^dim is large and N is moderate.
    """
    basis_mat = _as_basis_list(basis_mat)
    return [B @ B.T for B in basis_mat]


def mode_gram_from_basis(basis_mat, row_grams, mode):
    """
    Compute G_mode = A_(mode) A_(mode)^T without forming the full tensor A.

    The full tensor is

        A[i1, ..., id] = (1/N) sum_l B_1[l, i1] ... B_d[l, id].

    For mode k,

        G_k[a, b]
        = (1/N^2) sum_{l,m} B_k[l,a] B_k[m,b]
          prod_{j != k} <B_j[l], B_j[m]>.

    Thus G_k is only n by n.
    """
    basis_mat = _as_basis_list(basis_mat)
    dim = len(basis_mat)
    N, n = basis_mat[0].shape

    W = np.ones((N, N), dtype=float)
    for j in range(dim):
        if j != mode:
            W *= row_grams[j]

    B = basis_mat[mode]
    G = (B.T @ W @ B) / (N ** 2)

    # Numerical symmetrization.
    G = 0.5 * (G + G.T)
    return G


def tucker_factors_from_basis(
    basis_mat,
    threshold=0.9999,
    rank_rule="energy",
    min_rank=1,
    verbose=True,
):
    """
    Compute Tucker factor matrices using implicit mode Gram matrices.

    This replaces the expensive part of HOSVD that unfolds the full tensor.
    It computes the left singular vectors of each unfolding from

        G_k = A_(k) A_(k)^T,

    but it never forms A or A_(k).
    """
    basis_mat = _as_basis_list(basis_mat)
    dim = len(basis_mat)

    row_grams = compute_basis_row_grams(basis_mat)

    mat_list = []
    ranks = []
    singular_values_list = []

    for mode in range(dim):
        G = mode_gram_from_basis(basis_mat, row_grams, mode)

        eigvals, U = eigh(G)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        U = U[:, idx]

        eigvals = np.maximum(eigvals, 0.0)
        S = np.sqrt(eigvals)

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

    return mat_list, ranks, singular_values_list


def build_core_from_basis(basis_mat, mat_list):
    """
    Build the Tucker core without forming the full tensor.

    If U_j = mat_list[j], then

        core[a1, ..., ad]
        = (1/N) sum_l (B_1[l] U_1)[a1] ... (B_d[l] U_d)[ad].

    The cost depends on the Tucker ranks r_j, not on n^dim.
    """
    basis_mat = _as_basis_list(basis_mat)
    dim = len(basis_mat)
    N = basis_mat[0].shape[0]

    projected_basis = []
    for j in range(dim):
        U = np.asarray(mat_list[j], dtype=float)
        if U.ndim != 2:
            raise ValueError("Each Tucker factor must be 2D.")
        if U.shape[0] != basis_mat[j].shape[1]:
            raise ValueError("mat_list[j].shape[0] must equal n.")
        projected_basis.append(basis_mat[j] @ U)

    rank_labels = "abcdefghijklmnopqrstuvwxyABCDEFGHIJKLMNOPQRSTUVWXY"
    if dim > len(rank_labels):
        raise ValueError("dim is too large for this einsum construction.")

    input_terms = ["z" + rank_labels[j] for j in range(dim)]
    output_term = "".join(rank_labels[j] for j in range(dim))
    einsum_str = ",".join(input_terms) + "->" + output_term

    core = np.einsum(einsum_str, *projected_basis, optimize=True) / N
    return core


def tucker_implicit_from_basis(
    basis_mat,
    threshold=0.9999,
    rank_rule="energy",
    min_rank=1,
    verbose=True,
):
    """
    Compute an HOSVD-like Tucker decomposition without forming the full tensor.
    """
    basis_mat = _as_basis_list(basis_mat)

    mat_list, ranks, singular_values_list = tucker_factors_from_basis(
        basis_mat,
        threshold=threshold,
        rank_rule=rank_rule,
        min_rank=min_rank,
        verbose=verbose,
    )

    core = build_core_from_basis(basis_mat, mat_list)

    return core, mat_list, ranks, singular_values_list


class Tucker:
    """
    Tucker density estimator using implicit HOSVD.

    This version does not build the full tensor of size n^dim.
    It computes the mode factor matrices from n by n Gram matrices and then
    builds only the compressed Tucker core of size r_1 x ... x r_dim.
    """

    def __init__(
        self,
        n,
        X_train,
        max_iterate=0,
        threshold=0.9999,
        rank_rule="energy",
        max_memory_gb=None,
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

        if self.verbose:
            full_tensor_gib = (self.n ** self.dim) * 8 / (1024 ** 3)
            print(f"Not building full tensor of shape ({self.n},)^{self.dim}.")
            print(f"Avoided full tensor memory: about {full_tensor_gib:.6f} GiB")
            print("Computing Tucker factors from implicit Gram matrices.")

        self.core, self.mat_list, self.ranks, self.singular_values_list = (
            tucker_implicit_from_basis(
                self.basis_mat,
                threshold=self.threshold,
                rank_rule=self.rank_rule,
                min_rank=self.min_rank,
                verbose=self.verbose,
            )
        )

        if self.verbose:
            print("Tucker ranks:", self.ranks)
            print("Core shape:", self.core.shape)

        self.polynomial = polynomial(self.n, 1)
        self.sampler = Legendre_Sampler(self.n, self.polynomial.Linvt)

    def predict(self, X_test):
        """
        Evaluate the fitted density at test points.
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
    rng = np.random.default_rng(123)

    N = 200
    dim = 6
    n = 20

    X_train = rng.normal(size=(N, dim))

    model = Tucker(
        n=n,
        X_train=X_train,
        threshold=0.9999,
        rank_rule="energy",
        min_rank=1,
        verbose=True,
    )

    X_test = rng.normal(size=(10, dim))
    y_pred = model.predict(X_test)

    print("Prediction shape:", y_pred.shape)
    print("Predictions:", y_pred)
