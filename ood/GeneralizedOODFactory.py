import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedOODFactory(nn.Module):
    def __init__(self, tau=0.04, lambda_oe=0.5, gradnorm_gamma=0.01):
        super().__init__()
        self.tau = tau  # 3.1.1 LogitNorm Temperature
        self.lambda_oe = lambda_oe  # 3.1.2 OE Weight
        self.gamma = gradnorm_gamma  # 3.1.3 GradNorm Scaling

    def logit_norm_loss(self, logits, target):
        """Implementation of 3.1.1 LogitNorm"""
        norms = torch.norm(logits, p=2, dim=-1, keepdim=True) + 1e-7
        normed_logits = (logits / norms) / self.tau
        return F.cross_entropy(normed_logits, target)

    def outlier_exposure_loss(self, oe_logits):
        """Implementation of 3.1.2 Outlier Exposure"""
        # Pushes OOD data toward uniform distribution
        return -(oe_logits.mean(1) - torch.logsumexp(oe_logits, dim=1)).mean()

    def get_gradnorm_weight(self, features, head_module):
        """Implementation of 3.1.3 GradNorm Indicator"""
        # Ensure gradients can be computed for this post-hoc step
        with torch.enable_grad():
            features = features.detach().requires_grad_(True)
            # Forward pass through the specific task head
            logits = head_module(features)

            # Uniform targets for info-entropy gradient
            targets = torch.ones_like(logits) / logits.shape[1]
            loss = torch.mean(torch.sum(-targets * F.log_softmax(logits, dim=-1), dim=-1))

            # Extract gradients w.r.t features
            grads = torch.autograd.grad(loss, features)[0]
            score = torch.norm(grads, p=1, dim=-1)

            # Sigmoid gating: High GradNorm = Current Task (Online Expert)
            return torch.sigmoid(score * self.gamma).unsqueeze(-1)

    def compute_training_loss(self, id_logits, id_targets, oe_logits=None):
        """Wraps 3.1.1 and 3.1.2 for the Learning phase"""
        loss = self.logit_norm_loss(id_logits, id_targets)

        if oe_logits is not None:
            loss += self.lambda_oe * self.outlier_exposure_loss(oe_logits)

        return loss

    def compute_ensemble(self, logits_on, feat_on, logits_off, head_on, *, return_logits=False):
        """
        约定：
          logits_on  : (B, Cg)  全局类别 logits（未softmax）
          logits_off : (B, Cg)  全局类别 logits（未softmax）
          feat_on    : (B, D')  向量特征（用于计算 w）
          head_on    : nn.Module 或可调用对象（建议传当前 task 的线性层，便于 GradNorm）
        """
        import torch

        # 1) 计算权重 w（标量或(B,)），并广播为 (B, 1)
        w = self.get_gradnorm_weight(feat_on, head_on)
        if not torch.is_tensor(w):
            w = torch.tensor(w, device=logits_on.device, dtype=logits_on.dtype)
        w = w.to(device=logits_on.device, dtype=logits_on.dtype)
        if w.ndim == 0:
            pass
        elif w.ndim == 1:
            w = w.view(-1, 1)
        else:
            w = w.flatten().view(-1, 1)
        # 可选：数值安全
        w = w.clamp(0.0, 1.0)

        # 2) 形状检查
        if logits_on.ndim != 2 or logits_off.ndim != 2:
            raise ValueError(f"Expect 2D logits (B, C). Got {logits_on.shape=} and {logits_off.shape=}.")
        if logits_on.size(1) != logits_off.size(1):
            raise ValueError(
                f"Class-dim mismatch: online={logits_on.size(1)}, offline={logits_off.size(1)}."
            )

        # 3) 概率加权
        p_on = torch.softmax(logits_on, dim=1)
        p_off = torch.softmax(logits_off, dim=1)
        p = w * p_on + (1.0 - w) * p_off

        if return_logits:
            return torch.log(p.clamp_min(1e-12))
        return p