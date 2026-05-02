import numpy as np
from scipy.linalg import eigh

from local_basis import generate_basis_mat, polynomial
from sampling_local import Legendre_Sampler


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def estimate_tensor_memory_gb(shape, dtype=np.float64):
    """
    Estimate memory needed to store an array with the given shape.
    """
    numel = int(np.prod(shape))
    return numel * np.dtype(dtype).itemsize / (1024 ** 3)



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
        Singular values of a mode unfolding.
    threshold : float or int
        Threshold used by the selected rule.
    rank_rule : str
        One of "energy", "relative", "absolute", "eigen_absolute", or "fixed".
    min_rank : int
        Minimum rank to keep.
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
    rank = max(rank, int(min_rank))
    rank = min(rank, len(s))
    return rank



def _descending_eigh(G):
    """
    Eigen-decomposition of a symmetric matrix, sorted in descending order.

    Returns
    -------
    eigvals : ndarray
        Eigenvalues in descending order, clipped below by 0 only when forming
        singular values elsewhere.
    eigvecs : ndarray
        Corresponding eigenvectors.
    """
    G = 0.5 * (G + G.T)
    eigvals, eigvecs = eigh(G, check_finite=False)
    idx = np.argsort(eigvals)[::-1]
    return eigvals[idx], eigvecs[:, idx]



def _validate_basis_mat(basis_mat):
    """
    Convert basis_mat to a list of arrays with shape (N, n).
    """
    basis_mat = [np.asarray(B, dtype=float) for B in basis_mat]

    if len(basis_mat) == 0:
        raise ValueError("basis_mat must contain at least one mode.")

    N, n = basis_mat[0].shape
    for j, B in enumerate(basis_mat):
        if B.ndim != 2:
            raise ValueError(f"basis_mat[{j}] must be 2D.")
        if B.shape != (N, n):
            raise ValueError("All basis matrices must have the same shape (N, n).")

    return basis_mat, N, n, len(basis_mat)


# -----------------------------------------------------------------------------
# Implicit HOSVD initialization and HOOI updates
# -----------------------------------------------------------------------------


def implicit_mode_gram(basis_mat, mode, mat_list=None):
    """
    Compute the small mode Gram matrix without building the full tensor.

    Let

        A = (1/N) sum_l B_1[l] \\otimes ... \\otimes B_d[l].

    If mat_list is None, this computes

        G_mode = A_(mode) A_(mode)^T, shape (n, n).

    If mat_list is given, this computes the HOOI update Gram matrix after
    projecting all other modes onto their current Tucker factors:

        G_mode = Y_(mode) Y_(mode)^T,

    where

        Y = A x_j U_j^T for all j != mode.

    The formula is

        G_mode[a,b]
        = (1/N^2) sum_{l,m} B_mode[l,a] B_mode[m,b]
          prod_{j != mode} <C_j[l], C_j[m]>,

    where C_j = B_j if mat_list is None, and C_j = B_j U_j otherwise.
    """
    basis_mat, N, n, dim = _validate_basis_mat(basis_mat)

    if mode < 0 or mode >= dim:
        raise ValueError("mode is out of range.")

    if mat_list is not None and len(mat_list) != dim:
        raise ValueError("mat_list must have length dim.")

    W = np.ones((N, N), dtype=float)

    for j in range(dim):
        if j == mode:
            continue

        if mat_list is None:
            Cj = basis_mat[j]
        else:
            Cj = basis_mat[j] @ mat_list[j]

        W *= Cj @ Cj.T

    Bk = basis_mat[mode]
    G = Bk.T @ (W @ Bk)
    G /= float(N * N)
    G = 0.5 * (G + G.T)

    return G



def implicit_hosvd_initialization(
    basis_mat,
    threshold=0.9999,
    rank_rule="energy",
    min_rank=1,
    ranks=None,
    verbose=True,
):
    """
    Initialize Tucker factors using implicit HOSVD.

    No full tensor is formed. Each mode is initialized by the leading
    eigenvectors of the corresponding implicit mode Gram matrix.
    """
    basis_mat, N, n, dim = _validate_basis_mat(basis_mat)

    if ranks is not None:
        if len(ranks) != dim:
            raise ValueError("ranks must have length dim.")
        ranks = [int(r) for r in ranks]
        for r in ranks:
            if r < 1 or r > n:
                raise ValueError("Each rank must be between 1 and n.")

    mat_list = []
    rank_list = []
    singular_values_list = []

    for mode in range(dim):
        G = implicit_mode_gram(basis_mat, mode, mat_list=None)
        eigvals, eigvecs = _descending_eigh(G)
        singular_values = np.sqrt(np.maximum(eigvals, 0.0))

        if ranks is None:
            rank_k = choose_rank_from_singular_values(
                singular_values,
                threshold=threshold,
                rank_rule=rank_rule,
                min_rank=min_rank,
            )
        else:
            rank_k = ranks[mode]

        Uk = eigvecs[:, :rank_k]

        mat_list.append(Uk)
        rank_list.append(rank_k)
        singular_values_list.append(singular_values)

        if verbose:
            print(f"initial mode {mode}: singular values = {singular_values}")
            print(f"initial mode {mode}: rank = {rank_k}")

    return mat_list, rank_list, singular_values_list



def projected_norm_sq(basis_mat, mat_list):
    """
    Compute || A x_1 U_1^T ... x_d U_d^T ||_F^2 implicitly.

    This is the HOOI objective. It avoids building the core.
    """
    basis_mat, N, n, dim = _validate_basis_mat(basis_mat)

    if len(mat_list) != dim:
        raise ValueError("mat_list must have length dim.")

    W = np.ones((N, N), dtype=float)

    for j in range(dim):
        Cj = basis_mat[j] @ mat_list[j]
        W *= Cj @ Cj.T

    val = np.sum(W) / float(N * N)
    return float(max(val, 0.0))



def build_core_from_factors(basis_mat, mat_list, max_core_memory_gb=8.0):
    """
    Build the Tucker core from the projected basis matrices.

    core[a1, ..., ad]
    = (1/N) sum_l (B_1[l] U_1)[a1] ... (B_d[l] U_d)[ad].
    """
    basis_mat, N, n, dim = _validate_basis_mat(basis_mat)

    if len(mat_list) != dim:
        raise ValueError("mat_list must have length dim.")

    projected = []
    ranks = []
    for j in range(dim):
        Cj = basis_mat[j] @ mat_list[j]
        projected.append(Cj)
        ranks.append(Cj.shape[1])

    memory_gb = estimate_tensor_memory_gb(ranks)
    if memory_gb > max_core_memory_gb:
        raise MemoryError(
            f"The Tucker core has shape {tuple(ranks)} and needs about "
            f"{memory_gb:.2f} GB, which exceeds max_core_memory_gb="
            f"{max_core_memory_gb}. Use smaller ranks."
        )

    rank_labels = "abcdefghijklmnopqrstuvwxyABCDEFGHIJKLMNOPQRSTUVWXY"
    if dim > len(rank_labels):
        raise ValueError("dim is too large for this einsum construction.")

    input_terms = ["z" + rank_labels[j] for j in range(dim)]
    output_term = "".join(rank_labels[j] for j in range(dim))
    einsum_str = ",".join(input_terms) + "->" + output_term

    core = np.einsum(einsum_str, *projected, optimize=True) / float(N)
    return core



def implicit_hooi(
    basis_mat,
    threshold=0.9999,
    rank_rule="energy",
    ranks=None,
    min_rank=1,
    max_iterate=10,
    tol=1e-8,
    verbose=True,
):
    """
    HOOI for the tensor

        A = (1/N) sum_l B_1[l] \\otimes ... \\otimes B_d[l]

    without building A.

    The ranks are fixed during the HOOI iterations. If ranks is None, they are
    chosen once using implicit HOSVD and the given rank_rule.
    """
    basis_mat, N, n, dim = _validate_basis_mat(basis_mat)

    mat_list, rank_list, init_singular_values_list = implicit_hosvd_initialization(
        basis_mat,
        threshold=threshold,
        rank_rule=rank_rule,
        min_rank=min_rank,
        ranks=ranks,
        verbose=verbose,
    )

    max_iterate = int(max_iterate)
    if max_iterate < 0:
        raise ValueError("max_iterate must be nonnegative.")

    history = []
    prev_obj = projected_norm_sq(basis_mat, mat_list)
    history.append(prev_obj)

    if verbose:
        print(f"HOOI initial projected norm^2 = {prev_obj:.12e}")

    singular_values_list = init_singular_values_list

    for it in range(max_iterate):
        singular_values_list = []

        for mode in range(dim):
            G = implicit_mode_gram(basis_mat, mode, mat_list=mat_list)
            eigvals, eigvecs = _descending_eigh(G)
            singular_values = np.sqrt(np.maximum(eigvals, 0.0))
            singular_values_list.append(singular_values)

            rank_k = rank_list[mode]
            mat_list[mode] = eigvecs[:, :rank_k]

        obj = projected_norm_sq(basis_mat, mat_list)
        history.append(obj)

        rel_change = abs(obj - prev_obj) / max(abs(prev_obj), 1e-14)

        if verbose:
            print(
                f"HOOI iter {it + 1}: projected norm^2 = {obj:.12e}, "
                f"relative change = {rel_change:.3e}"
            )

        if rel_change < tol:
            if verbose:
                print(f"HOOI converged at iteration {it + 1}.")
            break

        prev_obj = obj

    return mat_list, rank_list, singular_values_list, history


# -----------------------------------------------------------------------------
# Estimator class
# -----------------------------------------------------------------------------


class Tucker:
    """
    Tucker density estimator using implicit HOOI.

    This version never builds the full tensor of size n^dim. It first obtains
    an implicit HOSVD initialization, then runs HOOI iterations. Each HOOI mode
    update diagonalizes an n by n Gram matrix.

    Parameters
    ----------
    n : int
        Number of basis functions per coordinate.
    X_train : ndarray, shape (N, dim)
        Training data.
    max_iterate : int
        Number of HOOI sweeps. Use 0 for implicit HOSVD only.
    threshold : float or int
        Rank threshold. If rank_rule="fixed", this is the common fixed rank.
    rank_rule : str
        One of "energy", "relative", "absolute", "eigen_absolute", or "fixed".
    ranks : list of int or None
        Optional mode-specific ranks. If provided, this overrides threshold and
        rank_rule for rank selection.
    min_rank : int
        Minimum rank if ranks is None.
    tol : float
        Stopping tolerance for relative improvement of the projected norm.
    max_core_memory_gb : float
        Safety limit for storing the final Tucker core.
    verbose : bool
        Whether to print diagnostic information.
    """

    def __init__(
        self,
        n,
        X_train,
        max_iterate=10,
        threshold=0.9999,
        rank_rule="energy",
        ranks=None,
        min_rank=1,
        tol=1e-8,
        max_core_memory_gb=8.0,
        verbose=True,
        # Kept for compatibility with your previous class. It is not used,
        # because this version does not build the full tensor.
        max_memory_gb=None,
    ):
        self.n = int(n)
        self.threshold = threshold
        self.rank_rule = rank_rule
        self.min_rank = int(min_rank)
        self.max_iterate = int(max_iterate)
        self.tol = float(tol)
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
            full_memory_gb = estimate_tensor_memory_gb((self.n,) * self.dim)
            print(f"Implicit HOOI: no full tensor is built.")
            print(f"Avoided full tensor shape: ({self.n},)^{self.dim}")
            print(f"Avoided full tensor memory: {full_memory_gb:.6f} GB")

        self.mat_list, self.ranks, self.singular_values_list, self.history = implicit_hooi(
            self.basis_mat,
            threshold=self.threshold,
            rank_rule=self.rank_rule,
            ranks=ranks,
            min_rank=self.min_rank,
            max_iterate=self.max_iterate,
            tol=self.tol,
            verbose=self.verbose,
        )

        if self.verbose:
            print("Final Tucker ranks:", self.ranks)

        self.core = build_core_from_factors(
            self.basis_mat,
            self.mat_list,
            max_core_memory_gb=max_core_memory_gb,
        )

        if self.verbose:
            print("Final core shape:", self.core.shape)
            print("Final core memory:", f"{estimate_tensor_memory_gb(self.core.shape):.6f} GB")

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
    # Minimal example.
    # This requires your local_basis.py and sampling_local.py files to be
    # available in the same Python environment.
    rng = np.random.default_rng(123)

    N = 200
    dim = 6
    n = 20

    X_train = rng.normal(size=(N, dim))

    model = Tucker(
        n=n,
        X_train=X_train,
        max_iterate=5,
        threshold=8,
        rank_rule="fixed",
        verbose=True,
    )

    X_test = rng.normal(size=(10, dim))
    y_pred = model.predict(X_test)

    print("Prediction shape:", y_pred.shape)
    print("Predictions:", y_pred)
