import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------
# Utilities
# ----------------------------
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
# MLP
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

# ----------------------------
# Unconditional VAE
# ----------------------------
class VAE(nn.Module):
    """
    Encoder: q(z | x)
    Decoder: p(x | z)   (outputs mean; noise sigma learned as global diag)
    """
    def __init__(self, data_dim, z_dim=10, enc_hidden=(128, 128), dec_hidden=(128, 128)):
        super().__init__()
        self.data_dim = data_dim
        self.z_dim = z_dim

        # encoder maps x -> (mu_z, logvar_z)
        self.enc = MLP(in_dim=data_dim, hidden_dims=enc_hidden, out_dim=2 * z_dim)

        # decoder maps z -> x_mean
        self.dec = MLP(in_dim=z_dim, hidden_dims=dec_hidden, out_dim=data_dim)

        # global diagonal noise for p(x|z) = N(x_mean, diag(sigma^2))
        self.log_sigma = nn.Parameter(torch.zeros(data_dim))

    def encode(self, x):
        out = self.enc(x)
        mu, logvar = out[..., :self.z_dim], out[..., self.z_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(self, z):
        x_mean = self.dec(z)
        return x_mean

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_mean = self.decode(z)
        return x_mean, mu, logvar

def vae_loss(x, x_mean, mu, logvar, log_sigma, beta=1.0):
    """
    Negative ELBO (up to constants):
      -E_q log p(x|z) + beta * KL(q(z|x) || N(0,I))
    with p(x|z)=N(x_mean, diag(sigma^2)), sigma=exp(log_sigma).
    """
    sigma2 = torch.exp(2.0 * log_sigma)  # (data_dim,)
    recon = 0.5 * (((x - x_mean) ** 2) / sigma2 + 2.0 * log_sigma).sum(dim=-1).mean()
    kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    return recon + beta * kl, recon.detach(), kl.detach()

# ----------------------------
# Main function: train + sample
# ----------------------------
def unconditional_samples_vae(
    X_train,
    N_sample,
    z_dim=10,
    enc_hidden=(128, 128),
    dec_hidden=(128, 128),
    epochs=300,
    batch_size=256,
    lr=1e-3,
    beta=1.0,
    device=None,
    verbose=False,
):
    """
    Train an unconditional VAE on X_train and sample from the fitted model.

    Parameters
    ----------
    X_train : array, shape (N, dim)
    N_sample : int
        Number of samples to generate.

    Returns
    -------
    X_samp : array, shape (N_sample, dim)
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    N, dim = X_train.shape

    # standardize data
    mu_all, sd_all = _standardize_fit(X_train)
    Xz = _standardize_apply(X_train, mu_all, sd_all)

    # torch setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = TensorDataset(torch.from_numpy(Xz))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = VAE(
        data_dim=dim,
        z_dim=z_dim,
        enc_hidden=enc_hidden,
        dec_hidden=dec_hidden,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # train
    model.train()
    for ep in range(epochs):
        total = 0.0
        for (xb,) in dl:
            xb = xb.to(device)

            x_mean, mu_z, logvar_z = model(xb)
            loss, recon, kl = vae_loss(xb, x_mean, mu_z, logvar_z, model.log_sigma, beta=beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += loss.item() * xb.size(0)

        if verbose and ((ep + 1) % 50 == 0 or ep == 0 or ep == epochs - 1):
            print(f"epoch {ep+1:4d}/{epochs}, loss={total/N:.4f}")

    # unconditional sampling
    model.eval()
    with torch.no_grad():
        z = torch.randn((N_sample, z_dim), device=device)
        x_mean = model.decode(z)

        sigma = torch.exp(model.log_sigma).reshape(1, -1)
        x = x_mean + torch.randn_like(x_mean) * sigma

        x = x.cpu().numpy()

    # invert standardization
    X_out = _standardize_invert(x, mu_all, sd_all)
    return X_out