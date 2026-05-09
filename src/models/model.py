"""
Model 4: Mixture of 4D Gaussians with Full Covariance (+ ablation switches).

At each brain grid location, G Gaussians form a local mixture model, allowing
non-symmetric modulation of the electrical field over time. Each Gaussian has:
  - spatial center:   FIXED to grid position p_i (default) or learnable
  - temporal center:  learnable (mu_t)
  - 4D covariance:    parameterized via Cholesky factor of the precision matrix
  - amplitude:        learnable

Ablation switches (see ABLATIONS.md): covariance_mode, offdiag_mode, center_mode,
ld_conditioning, mixture_assignment, forward_operator. All default to behaviour
matching the original "full" model so existing scripts run unchanged.

Backward-compat with the legacy `cov_ablation` arg: 'full' -> offdiag_mode='all',
'no_spacetime_coupling' -> 'spatial_only', 'diagonal_precision' -> 'none'.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .base import EEGSourceModelBase

logger = logging.getLogger(__name__)


# Map legacy cov_ablation values to the new offdiag_mode values
_COV_ABLATION_TO_OFFDIAG = {
    "full": "all",
    "no_spacetime_coupling": "spatial_only",
    "diagonal_precision": "none",
}


class EMAG(EEGSourceModelBase):
    """G Gaussians per active grid point, full 4D covariance (default), with configurable ablations."""

    def __init__(
        self,
        n_electrodes: int = 64,
        n_grid: int = 8000,
        grid_resolution: int = 20,
        sampling_rate: int = 250,
        n_ld_channels: int = 24,
        ld_feature_dim: int = 32,
        n_gaussians_per_point: int = 3,
        device: str = "cpu",
        electrode_positions=None,
        grid_type: str = "cube",
        grid_support: str = "volume",
        # ---- legacy single switch (kept for backward compat with existing scripts) ----
        cov_ablation: Optional[str] = None,
        # ---- new ablation switches ----
        covariance_mode: str = "full_4d",
        offdiag_mode: str = "all",
        center_mode: str = "fixed_grid",
        ld_conditioning: str = "per_grid",
        mixture_assignment: str = "uniform",
        active_point_ratio: float = 1.0,
        mixture_seed: int = 0,
        forward_operator: str = "gaussian_splat",
        learned_projection_rank: int = 0,
        leadfield_matrix: Optional[torch.Tensor] = None,
        # ---- local optimisation (preserved) ----
        inner_chunk: int = 10,
    ):
        super().__init__(
            n_electrodes=n_electrodes,
            n_grid=n_grid,
            grid_resolution=grid_resolution,
            sampling_rate=sampling_rate,
            n_ld_channels=n_ld_channels,
            ld_feature_dim=ld_feature_dim,
            device=device,
            electrode_positions=electrode_positions,
            grid_type=grid_type,
            grid_support=grid_support,
        )

        # Legacy cov_ablation -> offdiag_mode (only if user didn't explicitly set offdiag_mode)
        if cov_ablation is not None and cov_ablation in _COV_ABLATION_TO_OFFDIAG:
            offdiag_mode = _COV_ABLATION_TO_OFFDIAG[cov_ablation]

        self.cov_ablation = cov_ablation if cov_ablation is not None else "full"
        self.covariance_mode = covariance_mode
        self.offdiag_mode = offdiag_mode
        self.center_mode = center_mode
        self.ld_conditioning = ld_conditioning
        self.mixture_assignment = mixture_assignment
        self.active_point_ratio = float(active_point_ratio)
        self.mixture_seed = int(mixture_seed)
        self.forward_operator = forward_operator
        self.learned_projection_rank = int(learned_projection_rank)
        self.inner_chunk = int(inner_chunk)

        if covariance_mode not in ("full_4d", "spatial_3d_only", "diagonal_4d", "isotropic_scalar"):
            raise ValueError(f"Unknown covariance_mode: {covariance_mode}")
        if offdiag_mode not in ("all", "none", "spatial_only", "spacetime_only"):
            raise ValueError(f"Unknown offdiag_mode: {offdiag_mode}")
        if center_mode not in ("fixed_grid", "learnable"):
            raise ValueError(f"Unknown center_mode: {center_mode}")
        if ld_conditioning not in ("per_grid", "none", "global_scalar"):
            raise ValueError(f"Unknown ld_conditioning: {ld_conditioning}")
        if mixture_assignment not in ("uniform", "subset_random", "subset_attention"):
            raise ValueError(f"Unknown mixture_assignment: {mixture_assignment}")
        if forward_operator not in ("gaussian_splat", "learned_linear", "leadfield_matrix"):
            raise ValueError(f"Unknown forward_operator: {forward_operator}")

        if mixture_assignment == "subset_attention":
            logger.warning(
                "mixture_assignment='subset_attention' uses lightweight LD-dependent "
                "gating on top of subset_random selection (see ABLATIONS.md Ablation 3.2)."
            )

        G = n_gaussians_per_point
        self.n_gaussians_per_point = G

        N_full = self.n_grid
        if mixture_assignment == "uniform":
            N_active = N_full
            active_idx = np.arange(N_active, dtype=np.int64)
        else:
            N_active = max(1, int(round(self.active_point_ratio * N_full)))
            rng = np.random.default_rng(self.mixture_seed % (2**32))
            active_idx = np.sort(rng.choice(N_full, size=N_active, replace=False))

        self.register_buffer("active_grid_idx", torch.from_numpy(active_idx).long())
        self.n_active_grid = N_active
        N_total = N_active * G
        self.n_total = N_total

        grid_pts = self.brain_grid.get_grid_points().cpu().numpy()
        grid_active = torch.from_numpy(grid_pts[active_idx]).float()

        elec_pos = self.electrode_pos.cpu()
        if center_mode == "fixed_grid":
            spatial_diff = elec_pos.unsqueeze(1) - grid_active.unsqueeze(0)
            spatial_diff = spatial_diff.unsqueeze(2).expand(
                -1, -1, G, -1
            ).reshape(self.n_electrodes, N_total, 3)
            self.register_buffer("spatial_diff", spatial_diff)
            self.centers = None
        else:
            init_pts = grid_active.unsqueeze(1).expand(-1, G, -1).reshape(N_total, 3)
            self.centers = nn.Parameter(init_pts.clone())
            self.register_buffer("spatial_diff", torch.zeros(1))

        # --- Per-Gaussian learnable parameters ---
        self.amplitude = nn.Parameter(torch.zeros(N_total))
        self.mu_t = nn.Parameter(torch.rand(N_total) * 0.5)

        if covariance_mode == "isotropic_scalar":
            self.log_sigma_space = nn.Parameter(torch.full((N_total,), float(np.log(15.0))))
            self.log_sigma_time = nn.Parameter(torch.full((N_total,), float(np.log(0.1))))
            self.chol_log_diag = None
            self.chol_off_diag = None
        else:
            self.log_sigma_space = None
            self.log_sigma_time = None
            init_log_diag = torch.tensor(
                [np.log(1.0 / 15.0)] * 3 + [np.log(1.0 / 0.1)], dtype=torch.float32
            )
            self.chol_log_diag = nn.Parameter(
                init_log_diag.unsqueeze(0).expand(N_total, -1).clone()
            )
            self.chol_off_diag = nn.Parameter(torch.zeros(N_total, 6))

        # LD modulation
        mod_out = N_active
        if ld_conditioning == "global_scalar":
            mod_out = 1
        self.ld_modulator = nn.Sequential(
            nn.Linear(ld_feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, mod_out),
        )

        if mixture_assignment == "subset_attention":
            self.point_gate = nn.Sequential(
                nn.Linear(ld_feature_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, N_active),
                nn.Sigmoid(),
            )
        else:
            self.point_gate = None

        if forward_operator == "learned_linear":
            rank = self.learned_projection_rank
            if rank and rank > 0:
                self.proj_u = nn.Parameter(torch.randn(n_electrodes, rank) * 0.02)
                self.proj_v = nn.Parameter(torch.randn(rank, N_total) * 0.02)
                self.register_buffer("leadfield_buf", torch.empty(0))
                self.proj_w = None
            else:
                self.proj_w = nn.Parameter(torch.randn(n_electrodes, N_total) * 0.02)
                self.proj_u = None
                self.proj_v = None
                self.register_buffer("leadfield_buf", torch.empty(0))
        elif forward_operator == "leadfield_matrix":
            if leadfield_matrix is None:
                L = torch.zeros(n_electrodes, N_total)
            else:
                L = leadfield_matrix.float()
                if L.shape != (n_electrodes, N_total):
                    raise ValueError(
                        f"leadfield_matrix shape {tuple(L.shape)} != ({n_electrodes}, {N_total})"
                    )
            self.register_buffer("leadfield_buf", L)
            self.proj_u = self.proj_v = self.proj_w = None
        else:
            self.register_buffer("leadfield_buf", torch.empty(0))
            self.proj_u = self.proj_v = self.proj_w = None

        self.to(device)

    # ------------------------------------------------------------------ #
    # Forward helpers
    # ------------------------------------------------------------------ #
    def _spatial_dx_dy_dz(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.center_mode == "fixed_grid":
            dx = self.spatial_diff[:, :, 0]
            dy = self.spatial_diff[:, :, 1]
            dz = self.spatial_diff[:, :, 2]
        else:
            diff = self.electrode_pos.unsqueeze(1) - self.centers.unsqueeze(0)
            dx = diff[:, :, 0]
            dy = diff[:, :, 1]
            dz = diff[:, :, 2]
        return dx, dy, dz

    def _offdiag_masked(self, off: torch.Tensor) -> torch.Tensor:
        if self.offdiag_mode == "all":
            return off
        mask = torch.ones(6, device=off.device, dtype=off.dtype)
        if self.offdiag_mode == "none":
            mask *= 0.0
        elif self.offdiag_mode == "spatial_only":
            mask[3:6] = 0.0
        elif self.offdiag_mode == "spacetime_only":
            mask[0:3] = 0.0
        return off * mask.view(1, 6)

    def forward(self, t: torch.Tensor, x_ld: Optional[torch.Tensor] = None) -> torch.Tensor:
        ld_features = None
        if x_ld is not None and self.ld_conditioning != "none":
            ld_features = self.ld_encoder(x_ld.transpose(1, 2))  # (B, T, ld_feat)

        is_1d = t.dim() == 1
        if is_1d:
            t = t.unsqueeze(0)

        B, T = t.shape
        M = self.n_electrodes
        G = self.n_gaussians_per_point
        N_total = self.n_total
        device = t.device

        dx, dy, dz = self._spatial_dx_dy_dz()

        # Build A (spatial-only Mahalanobis), C (cross), D (temporal-only) per covariance_mode
        if self.covariance_mode == "isotropic_scalar":
            sigma_s = torch.exp(self.log_sigma_space).clamp(min=1e-6)
            sigma_t = torch.exp(self.log_sigma_time).clamp(min=1e-6)
            dist_sq = dx ** 2 + dy ** 2 + dz ** 2
            A = dist_sq / (sigma_s.view(1, -1) ** 2)
            W = torch.exp(-0.5 * A)
            C = torch.zeros(M, N_total, device=device, dtype=W.dtype)
            D = (1.0 / (sigma_t ** 2)).to(device=device, dtype=W.dtype)
        else:
            diag = torch.exp(self.chol_log_diag)
            off = self.chol_off_diag
            if self.covariance_mode == "diagonal_4d":
                off = off * 0.0
            else:
                off = self._offdiag_masked(off)
            if self.covariance_mode == "spatial_3d_only":
                off = off.clone()
                off[:, 3:6] = 0.0

            z0_s = diag[:, 0] * dx + off[:, 0] * dy + off[:, 1] * dz
            z1_s = diag[:, 1] * dy + off[:, 2] * dz
            z2_s = diag[:, 2] * dz

            A = z0_s ** 2 + z1_s ** 2 + z2_s ** 2
            C = z0_s * off[:, 3] + z1_s * off[:, 4] + z2_s * off[:, 5]
            D = off[:, 3] ** 2 + off[:, 4] ** 2 + off[:, 5] ** 2 + diag[:, 3] ** 2
            W = torch.exp(-0.5 * A)
            if self.covariance_mode == "spatial_3d_only":
                C = torch.zeros(M, N_total, device=device, dtype=W.dtype)
                D = (diag[:, 3] ** 2).to(device=device, dtype=W.dtype)

        t_norm = t.float() / self.sampling_rate

        # Amplitude: base + LD modulation
        base_amp = self.amplitude
        if self.ld_conditioning == "none" or ld_features is None:
            ld_mod = torch.zeros(B, T, N_total, device=device)
        elif self.ld_conditioning == "global_scalar":
            g = self.ld_modulator(ld_features)  # (B, T, 1)
            ld_mod = g.unsqueeze(2).expand(-1, -1, self.n_active_grid, G).reshape(B, T, N_total)
        else:
            ld_mod = self.ld_modulator(ld_features)  # (B, T, N_active)
            if self.point_gate is not None:
                gate = self.point_gate(ld_features)
                ld_mod = ld_mod * gate
            ld_mod = ld_mod.unsqueeze(-1).expand(-1, -1, -1, G).reshape(B, T, N_total)

        amp = base_amp + ld_mod  # (B, T, N_total)

        eeg = torch.zeros(B, T, M, device=device)
        chunk = min(self.inner_chunk, T)

        for ts in range(0, T, chunk):
            te = min(ts + chunk, T)
            Tc = te - ts
            tau = t_norm[:, ts:te].unsqueeze(-1) - self.mu_t  # (B, Tc, N_total)

            if self.forward_operator == "gaussian_splat":
                # Fused exponent: -0.5 * (A + 2*tau*C + tau^2 * D) = -0.5 * ||L^T d||^2 <= 0.
                # Computing as a single exp() avoids overflow that the split
                # form (W * cross * temporal_damp) can hit when tau*C is large.
                tau_e = tau.unsqueeze(2)  # (B, Tc, 1, N)
                if self.covariance_mode == "spatial_3d_only":
                    full_exp = (
                        -0.5 * A.unsqueeze(0).unsqueeze(0)
                        - 0.5 * (tau_e ** 2) * D.view(1, 1, 1, -1)
                    )
                else:
                    full_exp = (
                        -0.5 * A.unsqueeze(0).unsqueeze(0)
                        - tau_e * C.unsqueeze(0).unsqueeze(0)
                        - 0.5 * (tau_e ** 2) * D.view(1, 1, 1, -1)
                    )
                gauss_full = torch.exp(full_exp)  # (B, Tc, M, N), all <= 1
                eeg[:, ts:te, :] = torch.bmm(
                    gauss_full.reshape(B * Tc, M, N_total),
                    amp[:, ts:te, :].reshape(B * Tc, N_total, 1),
                ).reshape(B, Tc, M)
            elif self.forward_operator == "learned_linear":
                temporal_damp = torch.exp(-0.5 * tau ** 2 * D)
                eff_amp = amp[:, ts:te, :] * temporal_damp
                if self.proj_u is not None:
                    tmp = torch.matmul(eff_amp, self.proj_v.t())
                    eeg[:, ts:te, :] = torch.matmul(tmp, self.proj_u.t())
                else:
                    eeg[:, ts:te, :] = torch.matmul(eff_amp, self.proj_w.t())
            else:  # leadfield_matrix
                temporal_damp = torch.exp(-0.5 * tau ** 2 * D)
                eff_amp = amp[:, ts:te, :] * temporal_damp
                L = self.leadfield_buf
                eeg[:, ts:te, :] = torch.matmul(eff_amp, L.t())

        if is_1d:
            eeg = eeg.squeeze(0)
        return eeg

    def render_source(self, t, ld_features=None):
        raise NotImplementedError("EMAG uses a custom forward.")

    def get_model_info(self) -> dict:
        return {
            "model_name": f"Mixture of 4D Gaussians ({self.covariance_mode}, offdiag={self.offdiag_mode})",
            "n_parameters": self.get_parameter_count(),
            "covariance_mode": self.covariance_mode,
            "offdiag_mode": self.offdiag_mode,
            "center_mode": self.center_mode,
            "ld_conditioning": self.ld_conditioning,
            "mixture_assignment": self.mixture_assignment,
            "forward_operator": self.forward_operator,
            "n_grid": self.n_grid,
            "n_active_grid": self.n_active_grid,
            "n_gaussians_per_point": self.n_gaussians_per_point,
            "n_total_components": self.n_total,
            "inner_chunk": self.inner_chunk,
            "cov_ablation_legacy": self.cov_ablation,
        }

    def load_state_dict(self, state_dict, strict=True):
        """
        Backward-compat checkpoint loader.

        Accepts checkpoints from earlier model versions:
          (a) split-style: 'chol_off_diag_spatial' (N_total, 3) + 'chol_off_diag_temporal' (N_total, 3)
              -> concat into single 'chol_off_diag' (N_total, 6)
          (b) old single-style: already 'chol_off_diag' (N_total, 6) -> pass through
          (c) checkpoints saved before 'active_grid_idx' / 'leadfield_buf' buffers were
              introduced — these are deterministically built in __init__ from config
              (seed/forward_operator), so we backfill them from the live module's state.
        """
        if "chol_off_diag_spatial" in state_dict and "chol_off_diag_temporal" in state_dict \
                and "chol_off_diag" not in state_dict:
            sp = state_dict.pop("chol_off_diag_spatial")
            tp = state_dict.pop("chol_off_diag_temporal")
            state_dict["chol_off_diag"] = torch.cat([sp, tp], dim=1)

        # Backfill buffers that may be missing from older checkpoints.
        own = super().state_dict()
        for k in ("active_grid_idx", "leadfield_buf"):
            if k not in state_dict and k in own:
                state_dict[k] = own[k]

        return super().load_state_dict(state_dict, strict=strict)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_ld = 24
    model = EMAG(
        n_electrodes=64,
        grid_resolution=8,
        n_ld_channels=n_ld,
        n_gaussians_per_point=3,
        device=device,
        grid_type="sphere",
    )
    B, T = 2, 10
    t = torch.arange(T, device=device).float().unsqueeze(0).expand(B, -1)
    x_ld = torch.randn(B, n_ld, T, device=device)
    with torch.no_grad():
        eeg = model(t, x_ld=x_ld)
    assert eeg.shape == (B, T, 64)
    print("Forward OK", model.get_model_info())
