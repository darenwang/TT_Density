import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------
# Utilities
# ----------------------------
def _as_1d_float(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return x

def _standardize_fit(X, eps=1e-8):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.maximum(sd, eps)
    return mu.astype(np.float32), sd.astype(np.float32)

def _standardize_apply(X, mu, sd):
    return (X - mu) / sd

def _standardize_invert(Xz, mu, sd):
    return Xz * sd + mu

# ----------------------------
# Conditional VAE
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, act=nn.ReLU):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            layers.append(act())
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class CVAE(nn.Module):
    """
    Encoder: q(z | y, x)  (takes [y, x])
    Decoder: p(y | z, x)  (outputs mean; noise sigma learned as global diag)
    """
    def __init__(self, x_dim, y_dim, z_dim=10, enc_hidden=(128, 128), dec_hidden=(128, 128)):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        # encoder maps [y, x] -> (mu_z, logvar_z)
        self.enc = MLP(in_dim=y_dim + x_dim, hidden_dims=enc_hidden, out_dim=2 * z_dim)

        # decoder maps [z, x] -> y_mean
        self.dec = MLP(in_dim=z_dim + x_dim, hidden_dims=dec_hidden, out_dim=y_dim)

        # global diagonal noise for p(y|z,x) = N(y_mean, diag(sigma^2))
        self.log_sigma = nn.Parameter(torch.zeros(y_dim))

    def encode(self, y, x):
        h = torch.cat([y, x], dim=-1)
        out = self.enc(h)
        mu, logvar = out[..., :self.z_dim], out[..., self.z_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(self, z, x):
        h = torch.cat([z, x], dim=-1)
        y_mean = self.dec(h)
        return y_mean

    def forward(self, y, x):
        mu, logvar = self.encode(y, x)
        z = self.reparameterize(mu, logvar)
        y_mean = self.decode(z, x)
        return y_mean, mu, logvar

def cvae_loss(y, y_mean, mu, logvar, log_sigma, beta=1.0):
    """
    Negative ELBO (up to constants):
      -E_q log p(y|z,x) + beta * KL(q(z|y,x) || N(0,I))
    with p(y|z,x)=N(y_mean, diag(sigma^2)), sigma=exp(log_sigma).
    """
    # reconstruction NLL for diagonal Gaussian
    sigma2 = torch.exp(2.0 * log_sigma)  # (y_dim,)
    # broadcast over batch
    recon = 0.5 * (((y - y_mean) ** 2) / sigma2 + 2.0 * log_sigma).sum(dim=-1).mean()

    # KL(q || p)
    kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

    return recon + beta * kl, recon.detach(), kl.detach()

# ----------------------------
# Main function: train + sample
# ----------------------------
def conditional_samples_cvae(
    X_train,
    x_given,
    N_sample,
    idx_given=None,
    z_dim=10,
    enc_hidden=(128, 128),
    dec_hidden=(128, 128),
    epochs=300,
    batch_size=512,
    lr=1e-3,
    beta=1.0,
    seed=0,
    device=None,
    verbose=False,
):
    """
    Train a conditional VAE on X_train and sample X_unknown | X_given = x_given.

    Parameters
    ----------
    X_train : array, shape (N, dim)
    x_given : array, shape (m,) where m < dim
    idx_given : array of ints, shape (m,), optional
        Which coordinates are given. If None, assumes first m coords.
    Returns
    -------
    X_samp : array, shape (N_sample, dim)
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    N, dim = X_train.shape

    x_given = _as_1d_float(x_given)
    m = x_given.size
    if m >= dim:
        raise ValueError("Need len(x_given) < dim for conditional sampling.")
    if idx_given is None:
        idx_given = np.arange(m, dtype=int)
    else:
        idx_given = np.asarray(idx_given, dtype=int).reshape(-1)
        if idx_given.size != m:
            raise ValueError("idx_given and x_given must have the same length.")
        if np.any(idx_given < 0) or np.any(idx_given >= dim):
            raise ValueError("idx_given has out-of-range entries.")
        if len(np.unique(idx_given)) != m:
            raise ValueError("idx_given must not contain duplicates.")

    idx_given_set = set(idx_given.tolist())
    idx_unknown = np.array([j for j in range(dim) if j not in idx_given_set], dtype=int)
    x_dim = m
    y_dim = dim - m

    # standardize whole vector (helps training)
    mu_all, sd_all = _standardize_fit(X_train)
    Xz = _standardize_apply(X_train, mu_all, sd_all)

    X_cond = Xz[:, idx_given]    # (N, m)
    Y_tgt  = Xz[:, idx_unknown]  # (N, dim-m)

    # torch setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = TensorDataset(torch.from_numpy(Y_tgt), torch.from_numpy(X_cond))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = CVAE(x_dim=x_dim, y_dim=y_dim, z_dim=z_dim, enc_hidden=enc_hidden, dec_hidden=dec_hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(epochs):
        total = 0.0
        for yb, xb in dl:
            yb = yb.to(device)
            xb = xb.to(device)

            y_mean, mu_z, logvar_z = model(yb, xb)
            loss, recon, kl = cvae_loss(yb, y_mean, mu_z, logvar_z, model.log_sigma, beta=beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += loss.item() * yb.size(0)

        if verbose and ((ep + 1) % 50 == 0 or ep == 0 or ep == epochs - 1):
            print(f"epoch {ep+1:4d}/{epochs}, loss={total/N:.4f}")

    # sample conditional
    model.eval()
    with torch.no_grad():
        # standardize x_given
        x_full = np.zeros((dim,), dtype=np.float32)
        x_full[idx_given] = x_given
        x_full_z = _standardize_apply(x_full, mu_all, sd_all)
        xg = x_full_z[idx_given].reshape(1, -1)  # (1, m)
        xg = torch.from_numpy(np.repeat(xg, N_sample, axis=0)).to(device)

        z = torch.randn((N_sample, z_dim), device=device)
        y_mean = model.decode(z, xg)

        # add noise using learned sigma (optional but matches the model)
        sigma = torch.exp(model.log_sigma).reshape(1, -1)
        y = y_mean + torch.randn_like(y_mean) * sigma

        y = y.cpu().numpy()

    # stitch back to full vector in standardized space
    X_out_z = np.zeros((N_sample, dim), dtype=np.float32)
    X_out_z[:, idx_given] = x_full_z[idx_given][None, :]
    X_out_z[:, idx_unknown] = y

    # invert standardization
    X_out = _standardize_invert(X_out_z, mu_all, sd_all)
    return X_out


# ----------------------------
# Example
# ----------------------------
if __name__ == "__main__":
    N, dim = 5000, 20
    X_train = np.random.randn(N, dim).astype(np.float32)

    x_given = np.array([0.2, -1.1, 0.7], dtype=np.float32)  # m=3
    samples = conditional_samples_cvae(
        X_train,
        x_given=x_given,
        N_sample=1000,
        idx_given=None,     # assumes first 3 dims are given
        epochs=200,
        verbose=True
    )
    print(samples.shape)  # (1000, 20)
