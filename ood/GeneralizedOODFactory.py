import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedOODFactory(nn.Module):
    """
    OOD/uncertainty utilities for LAE-style inference-time expert gating.

    Features
    --------
    - Training-time:
        * LogitNorm loss for ID samples.
        * Outlier Exposure (OE) loss for OOD auxiliary data.
    - Inference-time:
        * Energy-based OOD scoring and gating.
        * GradNorm-based gating: KL(softmax(z/T) || uniform) gradient norm wrt features.
        * Two-stage gating (Energy -> GradNorm refinement for borderline cases).
        * Hybrid fusion (Energy + GradNorm).

    Notes
    -----
    * All comments, docstrings, and identifiers are in English-only for paper-ready code.
    """

    def __init__(
        self,
        tau: float = 0.04,
        lambda_oe: float = 0.5,
        gradnorm_gamma: float = 0.01,
        ood_T: float = 1.0,
        energy_eps: float = 1e-12,
        two_stage: bool = False,
        two_stage_low_q: float = 0.2,
        two_stage_high_q: float = 0.8,
        hybrid_alpha: float = 0.7,
    ):
        super().__init__()
        # Training-time hyperparameters
        self.tau = float(tau)                 # LogitNorm temperature
        self.lambda_oe = float(lambda_oe)     # OE weight
        self.gamma = float(gradnorm_gamma)    # GradNorm scaling

        # Inference-time hyperparameters
        self.ood_T = float(ood_T)             # Temperature used by Energy/GradNorm scoring
        self.energy_eps = float(energy_eps)
        self.two_stage = bool(two_stage)      # Enable Energy->GradNorm two-stage gating
        self.two_stage_low_q = float(two_stage_low_q)
        self.two_stage_high_q = float(two_stage_high_q)
        self.hybrid_alpha = float(hybrid_alpha)

        # EMA stats for Energy normalization (lightweight)
        self.register_buffer("_ema_e_mean", torch.tensor(0.0))
        self.register_buffer("_ema_e_var", torch.tensor(1.0))
        self._ema_momentum = 0.05

    # --------------------
    # Training-time losses
    # --------------------
    def logit_norm_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """LogitNorm (Wei et al., ICML'22): L2-normalize logits and apply temperature."""
        norms = torch.norm(logits, p=2, dim=-1, keepdim=True) + 1e-7
        normed_logits = (logits / norms) / self.tau
        return F.cross_entropy(normed_logits, target)

    def outlier_exposure_loss(self, oe_logits: torch.Tensor) -> torch.Tensor:
        """
        Outlier Exposure (Hendrycks et al., ICLR'19):
        Encourage uniform predictions on auxiliary OOD data.
        Equivalent to: - (mean(z) - logsumexp(z))
        """
        return -(oe_logits.mean(1) - torch.logsumexp(oe_logits, dim=1)).mean()

    def compute_training_loss(
        self,
        id_logits: torch.Tensor,
        id_targets: torch.Tensor,
        oe_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Wrapper for training-time objectives (ID LogitNorm + optional OE)."""
        loss = self.logit_norm_loss(id_logits, id_targets)
        if oe_logits is not None:
            loss = loss + self.lambda_oe * self.outlier_exposure_loss(oe_logits)
        return loss

    # --------------------
    # Inference-time: Energy
    # --------------------
    @torch.no_grad()
    def energy_score(self, logits: torch.Tensor, T: float | None = None) -> torch.Tensor:
        """
        E(x) = -T * logsumexp(z / T).
        Lower energy => more ID-like; higher energy => more OOD-like.
        """
        if T is None:
            T = self.ood_T
        E = -float(T) * torch.logsumexp(logits / float(T), dim=-1)
        E = torch.nan_to_num(E, nan=0.0, posinf=1e4, neginf=-1e4)
        return E

    @torch.no_grad()
    def _update_energy_ema(self, E: torch.Tensor) -> None:
        """Update EMA stats for Energy normalization."""
        m = self._ema_momentum
        mean = E.mean()
        var = E.var(unbiased=False) + 1e-12
        self._ema_e_mean = (1 - m) * self._ema_e_mean + m * mean
        self._ema_e_var = (1 - m) * self._ema_e_var + m * var

    @torch.no_grad()
    def energy_gate(
        self,
        logits_on: torch.Tensor,
        logits_off: torch.Tensor,
        T: float | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Energy-based gating: produce w in [0, 1] from online/offline energies.
        Intuition:
            If online is more ID-like (lower energy), we want a higher w; otherwise lower w.

        Implementation:
            E_on  = energy(logits_on), E_off = energy(logits_off)
            dE    = E_off - E_on
            w     = sigmoid(dE / s) with s = EMA std if normalize=True else 1.
        """
        if T is None:
            T = self.ood_T

        E_on = self.energy_score(logits_on, T=T)
        E_off = self.energy_score(logits_off, T=T)

        self._update_energy_ema(torch.cat([E_on, E_off], dim=0))

        dE = E_off - E_on
        if normalize:
            std = torch.sqrt(self._ema_e_var).clamp_min(1e-6)
            scale = std
        else:
            scale = torch.tensor(1.0, device=dE.device, dtype=dE.dtype)

        w = torch.sigmoid(dE / scale)
        return w.unsqueeze(-1)  # (B, 1)

    # --------------------
    # Inference-time: GradNorm
    # --------------------
    def get_gradnorm_weight(
        self,
        features: torch.Tensor,
        head_module: nn.Module,
        T: float = 1.0,
    ) -> torch.Tensor:
        """
        GradNorm-based gating:
            1) Compute KL(softmax(z/T) || uniform) on head(features),
            2) Take gradient wrt features,
            3) Score = ||grad||_2 / sqrt(D),
            4) w = sigmoid(gamma * Score).
        """
        was_training = head_module.training
        head_module.eval()

        with torch.enable_grad():
            features = features.detach().requires_grad_(True)
            logits = head_module(features)
            logp = F.log_softmax(logits / T, dim=-1)
            u = torch.full_like(logits, 1.0 / logits.size(1))
            loss = -(u * logp).sum(dim=-1).mean()  # mean KL(p||u)
            grads = torch.autograd.grad(
                loss, features, retain_graph=False, create_graph=False
            )[0]
            score = torch.norm(grads, p=2, dim=-1) / (grads.size(-1) ** 0.5 + 1e-12)
            w = torch.sigmoid(score * self.gamma).unsqueeze(-1)

        if was_training:
            head_module.train()
        return w

    # --------------------
    # Inference-time: Ensemble
    # --------------------
    def compute_ensemble(
        self,
        logits_on: torch.Tensor,     # (B, C)
        feat_on: torch.Tensor,       # (B, D')
        logits_off: torch.Tensor,    # (B, C)
        head_on: nn.Module,          # current-task classifier for GradNorm
        *,
        gate: str = "gradnorm",      # 'gradnorm' | 'energy' | 'hybrid'
        T: float | None = None,
        return_logits: bool = False,
        two_stage: bool | None = None,
        energy_quantiles: tuple[float, float] | None = None,
        hybrid_alpha: float | None = None,
    ) -> torch.Tensor:
        """
        Probability-space expert ensembling:
            p = w * softmax(logits_on) + (1 - w) * softmax(logits_off).

        Gating strategies:
            - 'energy':   use Energy-based w,
            - 'gradnorm': use GradNorm-based w,
            - 'hybrid':   w = alpha * w_energy + (1 - alpha) * w_gradnorm.

        Two-stage gating (if enabled):
            Use Energy to preselect "borderline" samples by quantiles, then refine those via GradNorm.
        """
        if T is None:
            T = self.ood_T
        if two_stage is None:
            two_stage = self.two_stage
        if hybrid_alpha is None:
            hybrid_alpha = self.hybrid_alpha

        if logits_on.ndim != 2 or logits_off.ndim != 2:
            raise ValueError(
                f"Expect 2D logits (B, C). Got {logits_on.shape=} and {logits_off.shape=}."
            )
        if logits_on.size(1) != logits_off.size(1):
            raise ValueError(
                f"Class-dim mismatch: online={logits_on.size(1)}, offline={logits_off.size(1)}."
            )

        device, dtype = logits_on.device, logits_on.dtype
        gate = gate.lower().strip()
        if gate not in ("gradnorm", "energy", "hybrid"):
            raise ValueError(f"Unsupported gate='{gate}'.")

        # Precompute energy gating (used by 'energy', 'hybrid', and 'two_stage')
        with torch.no_grad():
            w_energy = self.energy_gate(logits_on, logits_off, T=T, normalize=True)  # (B, 1)

        if gate == "energy":
            w = w_energy

        elif gate == "gradnorm":
            if two_stage:
                # Determine "borderline" region via energy quantiles on online expert
                if energy_quantiles is not None:
                    low_q, high_q = energy_quantiles
                else:
                    low_q, high_q = self.two_stage_low_q, self.two_stage_high_q

                with torch.no_grad():
                    E_on = self.energy_score(logits_on, T=T)  # (B,)
                    q_low = torch.quantile(E_on, torch.tensor(low_q, device=device))
                    q_high = torch.quantile(E_on, torch.tensor(high_q, device=device))
                    need_refine = (E_on > q_low) & (E_on < q_high)

                # Default to energy gating, refine borderline samples via GradNorm
                w = w_energy.clone()
                if need_refine.any():
                    idx = need_refine.nonzero(as_tuple=False).squeeze(-1)
                    w_refine = self.get_gradnorm_weight(feat_on[idx], head_on, T=T)  # (b', 1)
                    w[idx] = w_refine.to(device=device, dtype=dtype)
            else:
                w = self.get_gradnorm_weight(feat_on, head_on, T=T).to(device=device, dtype=dtype)

        else:  # 'hybrid'
            w_grad = self.get_gradnorm_weight(feat_on, head_on, T=T).to(device=device, dtype=dtype)
            a = torch.tensor(hybrid_alpha, device=device, dtype=dtype)
            w = a * w_energy + (1.0 - a) * w_grad

        # Safety
        w = torch.nan_to_num(w, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        # Probability-space mixing
        p_on = torch.softmax(logits_on, dim=1)
        p_off = torch.softmax(logits_off, dim=1)
        p = (w * p_on + (1.0 - w) * p_off).clamp_min(1e-12)

        if return_logits:
            return torch.log(p)
        return p