"""
Base classes and utilities for EEG source reconstruction models.
Includes positional encoding, leadfield construction, and abstract base classes.
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class FourierFeatures(nn.Module):
    """
    Sinusoidal positional encoding (Fourier features).
    
    For input tensor x of shape (..., D_in), outputs (..., 2*n_freqs*D_in) by:
    [sin(2^0*π*x), cos(2^0*π*x), sin(2^1*π*x), cos(2^1*π*x), ...]
    """
    
    def __init__(self, n_freqs: int = 10):
        """
        Args:
            n_freqs: Number of frequency bands per input dimension.
        """
        super().__init__()
        self.n_freqs = n_freqs
        # Pre-compute frequency bands: 2^0, 2^1, ..., 2^(n_freqs-1)
        freqs = torch.pow(2.0, torch.arange(n_freqs, dtype=torch.float32))
        self.register_buffer("freqs", (freqs * np.pi).clone().detach())  # shape: (n_freqs,)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., D_in)
        Returns:
            Encoded tensor of shape (..., D_in * 2 * n_freqs)
        """
        # x: (..., D_in)
        # freqs: (n_freqs,)
        # x_scaled: (..., D_in, n_freqs)
        x_scaled = x.unsqueeze(-1) * self.freqs  # (..., D_in, n_freqs)
        
        # Interleave sin and cos: [..., sin(f*x[0]), cos(f*x[0]), sin(f*x[1]), ...]
        encoded_sin = torch.sin(x_scaled)  # (..., D_in, n_freqs)
        encoded_cos = torch.cos(x_scaled)  # (..., D_in, n_freqs)
        
        # Stack and reshape: (..., D_in, n_freqs, 2) -> (..., D_in * n_freqs * 2)
        encoded = torch.stack([encoded_sin, encoded_cos], dim=-1)  # (..., D_in, n_freqs, 2)
        encoded = encoded.reshape(*x.shape[:-1], -1)  # (..., D_in * n_freqs * 2)
        
        return encoded


class BrainGrid:
    """
    Utility class for managing the brain grid voxels.
    Brain domain: [-90, 90]mm in each dimension.

    Supports two grid types:
      - 'cube':   Uniform cubic grid (GR^3 points). Original behaviour.
      - 'sphere': Uniform cubic grid filtered to a sphere of radius 90 mm,
                  keeping only points inside the brain volume (~52 % of cube).

    grid_support:
      - 'volume': 3D points (default for cube and sphere volume).
      - 'surface': For grid_type 'sphere' only — points on the sphere shell
        r = BRAIN_RADIUS_MM using a (theta, phi) lattice (grid_resolution^2 points).
    """

    BRAIN_RADIUS_MM = 90.0

    def __init__(self, grid_resolution: int = 20, device: str = "cpu",
                 grid_type: str = "cube", grid_support: str = "volume"):
        """
        Args:
            grid_resolution: Resolution per dimension (e.g., 20 -> 20x20x20 base grid).
            device: Torch device.
            grid_type: 'cube' (full cubic grid) or 'sphere' (filtered to brain sphere).
            grid_support: 'volume' or 'surface' (surface requires grid_type 'sphere').
        """
        if grid_type not in ("cube", "sphere"):
            raise ValueError(f"grid_type must be 'cube' or 'sphere', got '{grid_type}'")
        if grid_support not in ("volume", "surface"):
            raise ValueError(f"grid_support must be 'volume' or 'surface', got '{grid_support}'")
        if grid_support == "surface" and grid_type != "sphere":
            raise ValueError("grid_support='surface' is only supported with grid_type='sphere'")

        self.grid_resolution = grid_resolution
        self.grid_type = grid_type
        self.grid_support = grid_support
        self.device = device
        self.brain_domain = (-90.0, 90.0)  # mm
        self.domain_size = self.brain_domain[1] - self.brain_domain[0]  # 180 mm

        if grid_type == "sphere" and grid_support == "surface":
            # Spherical shell at fixed radius (2D manifold sampling)
            R = self.BRAIN_RADIUS_MM
            theta = np.linspace(1e-6, np.pi - 1e-6, grid_resolution, dtype=np.float64)
            phi = np.linspace(0.0, 2.0 * np.pi, grid_resolution, endpoint=False, dtype=np.float64)
            tt, pp = np.meshgrid(theta, phi, indexing="ij")
            st = np.sin(tt)
            x = R * st * np.cos(pp)
            y = R * st * np.sin(pp)
            z = R * np.cos(tt)
            grid_points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        else:
            coords = np.linspace(
                self.brain_domain[0],
                self.brain_domain[1],
                grid_resolution
            )
            grid = np.meshgrid(coords, coords, coords, indexing="ij")
            grid_points = np.stack(grid, axis=-1).reshape(-1, 3)  # (GR^3, 3)

            if grid_type == "sphere":
                # Keep only points inside a sphere of radius BRAIN_RADIUS_MM
                dist_from_origin = np.linalg.norm(grid_points, axis=1)
                mask = dist_from_origin <= self.BRAIN_RADIUS_MM
                grid_points = grid_points[mask]

        self.register_buffer = lambda name, tensor: setattr(
            self, name, tensor.to(device)
        )
        self.grid_points = torch.from_numpy(grid_points).float()
        self.n_grid = len(grid_points)

    def get_grid_points(self) -> torch.Tensor:
        """Returns grid points as (N_grid, 3) in mm."""
        return self.grid_points.to(self.device)

    def get_grid_center(self) -> torch.Tensor:
        """Returns the center of the brain grid: (0, 0, 0)."""
        return torch.tensor([0.0, 0.0, 0.0], device=self.device)


def create_leadfield(
    n_electrodes: int = 64,
    n_grid: int = 8000,
    seed: int = 42,
    device: str = "cpu",
    electrode_positions: Optional[np.ndarray] = None,
    grid_resolution: int = 20,
    grid_type: str = "cube",
    grid_support: str = "volume",
) -> torch.Tensor:
    """
    Creates a leadfield matrix mapping brain source density to EEG signals.
    
    If electrode_positions are provided, computes a physics-based leadfield
    using inverse-distance decay (1/r^2). Otherwise falls back to random.
    
    Args:
        n_electrodes: Number of EEG electrodes (M).
        n_grid: Number of brain grid points (N).
        seed: Random seed for reproducibility.
        device: Torch device.
        electrode_positions: (M, 3) array of electrode positions in mm.
        grid_resolution: Grid resolution per dimension.
        grid_type: 'cube' or 'sphere'.
    
    Returns:
        Leadfield matrix L of shape (M, N).
        Maps brain source density S(N,) -> EEG signals (M,).
    """
    if electrode_positions is not None:
        # Use BrainGrid so grid_type filtering is applied consistently
        bg = BrainGrid(
            grid_resolution, device="cpu", grid_type=grid_type, grid_support=grid_support
        )
        grid_points = bg.get_grid_points().numpy()  # (N, 3)
        
        # Distance matrix: (M, N)
        diff = electrode_positions[:, np.newaxis, :] - grid_points[np.newaxis, :, :]  # (M, N, 3)
        dist_sq = np.sum(diff ** 2, axis=-1)  # (M, N)
        
        # Inverse distance decay with regularization
        eps = 100.0  # mm^2, prevents singularity for electrodes near grid points
        L = 1.0 / (dist_sq + eps)
        
        # Row-normalize: each electrode's total pickup is unit norm
        # This ensures L @ S has the same scale as S, preventing gradient explosion
        L = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-8)
        
        return torch.from_numpy(L).float().to(device)
    
    # Fallback: random leadfield
    np.random.seed(seed)
    rank = min(n_electrodes // 2, n_grid // 10)
    L_low = np.random.randn(n_electrodes, rank) @ np.random.randn(rank, n_grid)
    L_noise = 0.1 * np.random.randn(n_electrodes, n_grid)
    L = L_low + L_noise
    L = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-8)
    
    return torch.from_numpy(L).float().to(device)


class EEGSourceModelBase(nn.Module, ABC):
    """
    Abstract base class for EEG source reconstruction models.
    
    All models must:
    1. Accept low-density (LD) EEG as conditioning input.
    2. Output per-source amplitude and spatial variance on brain grid.
    3. Compute HD EEG as sum of Gaussian field contributions at electrode positions.
    4. Support batch processing over time.
    
    Forward model:
        EEG[j,t] = sum_i amplitude_i(t) * exp(-||electrode_j - grid_i||^2 / (2 * variance_i(t)))
    """
    
    def __init__(
        self,
        n_electrodes: int = 64,
        n_grid: int = 8000,
        grid_resolution: int = 20,
        sampling_rate: int = 250,
        n_ld_channels: int = 24,
        ld_feature_dim: int = 32,
        device: str = "cpu",
        electrode_positions: Optional[np.ndarray] = None,
        grid_type: str = "cube",
        grid_support: str = "volume",
    ):
        """
        Args:
            n_electrodes: Number of HD EEG electrodes (M).
            n_grid: Number of brain grid points (N).
            grid_resolution: Grid resolution (e.g., 20 -> 20^3).
            sampling_rate: EEG sampling rate (Hz).
            n_ld_channels: Number of low-density EEG channels for conditioning.
            ld_feature_dim: Dimension of per-timestep features extracted from LD EEG.
            device: Torch device ("cpu" or "cuda").
            electrode_positions: (M, 3) array of electrode positions in mm for
                physics-based leadfield. If None, uses random leadfield.
            grid_type: 'cube' or 'sphere' (sphere filters to brain volume).
            grid_support: 'volume' or 'surface' (passed to BrainGrid).
        """
        super().__init__()
        self.n_electrodes = n_electrodes
        self.n_grid = n_grid
        self.grid_resolution = grid_resolution
        self.sampling_rate = sampling_rate
        self.n_ld_channels = n_ld_channels
        self.ld_feature_dim = ld_feature_dim
        self.device = device
        self.grid_type = grid_type
        self.grid_support = grid_support

        # Create brain grid
        self.brain_grid = BrainGrid(
            grid_resolution, device, grid_type=grid_type, grid_support=grid_support
        )
        self.n_grid = self.brain_grid.n_grid  # Update to actual grid size

        # Store electrode positions and pre-compute distances for Gaussian forward model
        if electrode_positions is not None:
            elec_pos = torch.from_numpy(np.asarray(electrode_positions)).float()
        else:
            np.random.seed(42)
            elec_pos = torch.randn(n_electrodes, 3) * 50.0
        self.register_buffer("electrode_pos", elec_pos)

        # Pre-compute squared distances: grid point → electrode (on CPU; .to(device) later)
        # dist_sq[j, i] = ||electrode_j - grid_i||^2
        grid_pts = self.brain_grid.get_grid_points().cpu()  # (N_grid, 3)
        dist_sq = torch.cdist(
            elec_pos.unsqueeze(0), grid_pts.unsqueeze(0)
        ).squeeze(0).pow(2)  # (M, N_grid)
        self.register_buffer("grid_electrode_dist_sq", dist_sq)

        # LD encoder: per-timestep LD channels → feature vector
        self.ld_encoder = nn.Sequential(
            nn.Linear(n_ld_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, ld_feature_dim),
            nn.ReLU(inplace=True),
        )
    
    @abstractmethod
    def render_source(self, t: torch.Tensor, ld_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render source amplitudes and spatial log-variances, conditioned on LD features.
        
        Args:
            t: Time indices, shape (B, T) or (T,).
            ld_features: Per-timestep LD features, shape (B, T, ld_feature_dim).
        
        Returns:
            amplitude: (B, T, N_grid) — signed source amplitude at each grid point.
            logvar: (B, T, N_grid) — log spatial variance (isotropic) of each source.
        """
        pass
    
    def forward(self, t: torch.Tensor, x_ld: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass: encode LD input, render sources, compute EEG via Gaussian contributions.
        
        EEG[j,t] = sum_i amplitude_i(t) * exp(-||electrode_j - grid_i||^2 / (2 * var_i(t)))
        
        Args:
            t: Time indices, shape (B, T) or (T,).
            x_ld: Low-density EEG, shape (B, n_ld, T). Required for conditioning.
        
        Returns:
            EEG signals, shape (B, T, M_electrodes) or (T, M_electrodes).
        """
        # Encode LD EEG into per-timestep features
        ld_features = None
        if x_ld is not None:
            ld_transposed = x_ld.transpose(1, 2)       # (B, T, n_ld)
            ld_features = self.ld_encoder(ld_transposed) # (B, T, ld_feature_dim)

        # Get per-source amplitude and log-variance
        amplitude, logvar = self.render_source(t, ld_features=ld_features)

        is_1d = (amplitude.dim() == 2)
        if is_1d:
            amplitude = amplitude.unsqueeze(0)
            logvar = logvar.unsqueeze(0)

        B, T, N = amplitude.shape
        M = self.n_electrodes
        dist_sq = self.grid_electrode_dist_sq  # (M, N)

        eeg = torch.zeros(B, T, M, device=amplitude.device)
        chunk = min(50, T)  # internal chunk limits peak memory for gauss_w (B,Tc,M,N)
        for ts in range(0, T, chunk):
            te = min(ts + chunk, T)
            Tc = te - ts

            amp = amplitude[:, ts:te, :]               # (B, Tc, N)
            var = torch.exp(logvar[:, ts:te, :])        # (B, Tc, N)

            # Gaussian weights: G[b,t,j,i] = exp(-dist_sq[j,i] / (2*var[b,t,i]))
            gauss_w = torch.exp(
                -dist_sq.unsqueeze(0).unsqueeze(0) / (2 * var.unsqueeze(2))
            )  # (B, Tc, M, N)

            # EEG = G @ amplitude  (batched matrix multiply)
            eeg[:, ts:te, :] = torch.bmm(
                gauss_w.reshape(B * Tc, M, N),
                amp.reshape(B * Tc, N, 1)
            ).reshape(B, Tc, M)

        if is_1d:
            eeg = eeg.squeeze(0)

        return eeg
    
    def get_parameter_count(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
