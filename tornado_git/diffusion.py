import numpy as np
import torch
import torch.nn as nn
from torchcfm.conditional_flow_matching import TargetConditionalFlowMatcher


class MLPVectorField(nn.Module):
    def __init__(self, dim, hidden_dims=(128, 128, 128)):
        super().__init__()
        layers = []
        in_dim = dim + 1

        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.SiLU())
            in_dim = h

        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t, x):
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        t = t.reshape(-1, 1)
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)


class TorchCFMTabularGenerator:
    def __init__(
        self,
        hidden_dims=(128, 128, 128),
        lr=1e-3,
        batch_size=1024,
        epochs=120,
        sigma=0.01,
        euler_steps=100,          # kept for backward compatibility
        standardize=True,
        device=None,
        seed=123,
        weight_decay=1e-5,
        grad_clip=1.0,
        use_ema=True,
        ema_decay=0.999,
        sample_solver="heun",     # "euler", "heun", or "rk4"
        sample_steps=64,
    ):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.sigma = sigma
        self.euler_steps = euler_steps
        self.standardize = standardize
        self.seed = seed

        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.sample_solver = sample_solver
        self.sample_steps = sample_steps

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.model = None
        self.flow_matcher = TargetConditionalFlowMatcher(sigma=sigma)
        self.dim = None
        self.x_mean = None
        self.x_std = None
        self.loss_history = []
        self.ema_state = None

        torch.manual_seed(seed)
        np.random.seed(seed)

    def _to_tensor(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        return X.float()

    def _fit_standardizer(self, X):
        self.x_mean = X.mean(dim=0, keepdim=True)
        self.x_std = X.std(dim=0, keepdim=True)
        self.x_std = torch.where(self.x_std < 1e-8, torch.ones_like(self.x_std), self.x_std)

    def _transform(self, X):
        if not self.standardize:
            return X
        mean = self.x_mean.to(X.device)
        std = self.x_std.to(X.device)
        return (X - mean) / std

    def _inverse_transform(self, X):
        if not self.standardize:
            return X
        mean = self.x_mean.to(X.device)
        std = self.x_std.to(X.device)
        return X * std + mean

    @torch.no_grad()
    def _init_ema(self):
        self.ema_state = {
            k: v.detach().clone()
            for k, v in self.model.state_dict().items()
        }

    @torch.no_grad()
    def _update_ema(self):
        if not self.use_ema:
            return
        if self.ema_state is None:
            self._init_ema()
            return

        d = self.ema_decay
        for k, v in self.model.state_dict().items():
            self.ema_state[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

    def _build_sampling_model(self):
        if (not self.use_ema) or (self.ema_state is None):
            self.model.eval()
            return self.model

        model = MLPVectorField(self.dim, self.hidden_dims).to(self.device)
        model.load_state_dict(self.ema_state, strict=True)
        model.eval()
        return model

    def fit(self, X_train, verbose=True):
        X_train = self._to_tensor(X_train)

        if X_train.ndim != 2:
            raise ValueError("X_train must have shape (n_samples, n_features).")

        self.dim = X_train.shape[1]

        if self.standardize:
            self._fit_standardizer(X_train)
            X_train = self._transform(X_train)

        X_train = X_train.to(self.device)

        self.model = MLPVectorField(self.dim, self.hidden_dims).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=self.lr * 0.05,
        )

        n = X_train.shape[0]

        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            n_batches = 0

            self.model.train()

            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                x1 = X_train[idx]
                x0 = torch.randn_like(x1)

                t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1)
                vt = self.model(t, xt)

                loss = ((vt - ut) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                optimizer.step()
                self._update_ema()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            self.loss_history.append(avg_loss)

            if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch + 1:4d}/{self.epochs}, "
                    f"loss = {avg_loss:.6f}, lr = {current_lr:.6e}"
                )

        return self

    @torch.no_grad()
    def sample(self, N, return_tensor=False, chunk_size=4096):
        if self.model is None:
            raise RuntimeError("Call fit(X_train) before sample(N).")

        model = self._build_sampling_model()
        out = []

        steps = self.sample_steps
        dt = 1.0 / steps

        for start in range(0, N, chunk_size):
            m = min(chunk_size, N - start)
            x = torch.randn(m, self.dim, device=self.device)

            for k in range(steps):
                t0 = torch.full((m,), k * dt, device=self.device)

                if self.sample_solver == "euler":
                    v = model(t0, x)
                    x = x + dt * v

                elif self.sample_solver == "heun":
                    k1 = model(t0, x)
                    x_pred = x + dt * k1
                    t1 = torch.full((m,), min((k + 1) * dt, 1.0), device=self.device)
                    k2 = model(t1, x_pred)
                    x = x + 0.5 * dt * (k1 + k2)

                elif self.sample_solver == "rk4":
                    t_half = torch.full((m,), k * dt + 0.5 * dt, device=self.device)
                    t1 = torch.full((m,), min((k + 1) * dt, 1.0), device=self.device)

                    k1 = model(t0, x)
                    k2 = model(t_half, x + 0.5 * dt * k1)
                    k3 = model(t_half, x + 0.5 * dt * k2)
                    k4 = model(t1, x + dt * k3)

                    x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

                else:
                    raise ValueError("sample_solver must be one of: 'euler', 'heun', 'rk4'")

            out.append(x)

        x = torch.cat(out, dim=0)
        x = self._inverse_transform(x)

        if return_tensor:
            return x
        return x.cpu().numpy()