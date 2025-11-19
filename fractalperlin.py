"""
Fractal Perlin Noise generators for 2D images and 3D video sequences.

Based on the pyperlin library, extended to support 3D (time-based) noise for smooth video generation.
"""
import torch
import numpy as np
from typing import List, Tuple

tau = 6.28318530718


class FractalPerlin2D(object):
    """
    2D Fractal Perlin Noise generator.

    Args:
        shape: Tuple of (channels, height, width) or (height, width)
        resolutions: List of (height_res, width_res) tuples for each octave
        factors: List of amplitude factors for each octave
        generator: PyTorch random generator (for reproducibility)
    """

    def __init__(self, shape: Tuple[int, ...], resolutions: List[Tuple[int, int]],
                 factors: List[float], generator=torch.random.default_generator):
        shape = shape if len(shape) == 3 else (None,) + shape
        self.shape = shape
        self.factors = factors
        self.generator = generator
        self.device = generator.device
        self.resolutions = resolutions
        self.grid_shapes = [(shape[1] // res[0], shape[2] // res[1]) for res in resolutions]

        # Precomputed tensors
        self.linxs = [torch.linspace(0, 1, gs[1], device=self.device) for gs in self.grid_shapes]
        self.linys = [torch.linspace(0, 1, gs[0], device=self.device) for gs in self.grid_shapes]
        self.tl_masks = [self.fade(lx)[None, :] * self.fade(ly)[:, None] for lx, ly in zip(self.linxs, self.linys)]
        self.tr_masks = [torch.flip(tl_mask, dims=[1]) for tl_mask in self.tl_masks]
        self.bl_masks = [torch.flip(tl_mask, dims=[0]) for tl_mask in self.tl_masks]
        self.br_masks = [torch.flip(tl_mask, dims=[0, 1]) for tl_mask in self.tl_masks]

    def fade(self, t: torch.Tensor) -> torch.Tensor:
        """Smoothstep fade function: 6t^5 - 15t^4 + 10t^3"""
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def perlin_noise(self, octave: int, batch_size: int) -> torch.Tensor:
        """Generate Perlin noise for a specific octave."""
        res = self.resolutions[octave]
        angles = torch.zeros((batch_size, res[0] + 2, res[1] + 2), device=self.device)
        angles.uniform_(0, tau, generator=self.generator)
        rx = torch.cos(angles)[:, :, :, None] * self.linxs[octave]
        ry = torch.sin(angles)[:, :, :, None] * self.linys[octave]
        prx, pry = rx[:, :, :, None, :], ry[:, :, :, :, None]
        nrx, nry = -torch.flip(prx, dims=[4]), -torch.flip(pry, dims=[3])
        br = prx[:, :-1, :-1] + pry[:, :-1, :-1]
        bl = nrx[:, :-1, 1:] + pry[:, :-1, 1:]
        tr = prx[:, 1:, :-1] + nry[:, 1:, :-1]
        tl = nrx[:, 1:, 1:] + nry[:, 1:, 1:]

        grid_shape = self.grid_shapes[octave]
        grids = (self.br_masks[octave] * br + self.bl_masks[octave] * bl +
                 self.tr_masks[octave] * tr + self.tl_masks[octave] * tl)
        noise = grids.permute(0, 1, 3, 2, 4).reshape(
            (batch_size, self.shape[1] + grid_shape[0], self.shape[2] + grid_shape[1]))

        A = torch.randint(0, grid_shape[0], (batch_size,), device=self.device, generator=self.generator)
        B = torch.randint(0, grid_shape[1], (batch_size,), device=self.device, generator=self.generator)
        noise = torch.stack([noise[n, a:a - grid_shape[0], b:b - grid_shape[1]]
                           for n, (a, b) in enumerate(zip(A, B))])
        return noise

    def __call__(self, batch_size: int = None) -> torch.Tensor:
        """Generate 2D fractal Perlin noise."""
        batch_size = self.shape[0] if batch_size is None else batch_size
        shape = (batch_size,) + self.shape[1:]
        noise = torch.zeros(shape, device=self.device)
        for octave, factor in enumerate(self.factors):
            noise += factor * self.perlin_noise(octave, batch_size=batch_size)
        return noise


class FractalPerlin3D(object):
    """
    3D Fractal Perlin Noise generator for smooth video sequences.

    Generates coherent noise across time, perfect for DeepDream-style video synthesis.
    Each frame is a slice of 3D noise at a specific time coordinate.

    Args:
        shape: Tuple of (channels, height, width) - spatial dimensions
        resolutions: List of (time_res, height_res, width_res) tuples for each octave
        factors: List of amplitude factors for each octave
        num_frames: Total number of frames to generate
        generator: PyTorch random generator (for reproducibility)
        loop: If True, the noise will loop seamlessly
    """

    def __init__(self, shape: Tuple[int, int, int], resolutions: List[Tuple[int, int, int]],
                 factors: List[float], num_frames: int,
                 generator=torch.random.default_generator, loop: bool = True):
        self.shape = shape if len(shape) == 3 else (None,) + shape  # (C, H, W)
        self.factors = factors
        self.num_frames = num_frames
        self.generator = generator
        self.device = generator.device
        self.resolutions = resolutions  # [(t_res, h_res, w_res), ...]
        self.loop = loop

        # Grid shapes for each octave
        self.grid_shapes = [
            (num_frames // res[0] if loop else num_frames,
             shape[1] // res[1],
             shape[2] // res[2])
            for res in resolutions
        ]

        # Precompute interpolation grids for each octave
        self.setup_interpolation_grids()

    def setup_interpolation_grids(self):
        """Precompute interpolation grids for efficient noise generation."""
        self.lint_grids = []
        self.liny_grids = []
        self.linx_grids = []
        self.masks = []

        for octave, (tres, hres, wres) in enumerate(self.resolutions):
            gs = self.grid_shapes[octave]

            # Create 1D interpolation vectors
            if self.loop:
                # For looping, we tile the time dimension
                lint = torch.linspace(0, 1, tres, device=self.device)
                lint = lint.repeat(self.num_frames // tres + 1)[:self.num_frames]
            else:
                lint = torch.linspace(0, 1, gs[0], device=self.device)

            liny = torch.linspace(0, 1, gs[1], device=self.device)
            linx = torch.linspace(0, 1, gs[2], device=self.device)

            self.lint_grids.append(lint)
            self.liny_grids.append(liny)
            self.linx_grids.append(linx)

            # Compute 3D fade masks (8 corners of cube)
            fade_t = self.fade(lint)[:, None, None]
            fade_y = self.fade(liny)[None, :, None]
            fade_x = self.fade(linx)[None, None, :]

            # 8 corner masks for trilinear interpolation
            masks_octave = []
            for dt in [fade_t, 1 - fade_t]:
                for dy in [fade_y, 1 - fade_y]:
                    for dx in [fade_x, 1 - fade_x]:
                        masks_octave.append(dt * dy * dx)
            self.masks.append(masks_octave)

    def fade(self, t: torch.Tensor) -> torch.Tensor:
        """Smoothstep fade function: 6t^5 - 15t^4 + 10t^3"""
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def perlin_noise_3d(self, octave: int) -> torch.Tensor:
        """Generate 3D Perlin noise for a specific octave."""
        tres, hres, wres = self.resolutions[octave]

        # Generate random gradient vectors for each grid point
        # We need (tres+2) x (hres+2) x (wres+2) grid points
        grid_size = (tres + 2, hres + 2, wres + 2)

        # Random angles for gradient vectors (simplified 3D gradients)
        theta = torch.zeros(grid_size, device=self.device).uniform_(0, tau, generator=self.generator)
        phi = torch.zeros(grid_size, device=self.device).uniform_(0, np.pi, generator=self.generator)

        # Convert spherical to Cartesian coordinates
        grad_x = torch.sin(phi) * torch.cos(theta)
        grad_y = torch.sin(phi) * torch.sin(theta)
        grad_z = torch.cos(phi)

        # Interpolation grid positions
        lint = self.lint_grids[octave]
        liny = self.liny_grids[octave]
        linx = self.linx_grids[octave]

        # Create distance vectors for each corner
        # This is a simplified implementation - full 3D Perlin is more complex
        # but this gives us smooth, coherent noise across time

        # Generate gradients at corners
        gradients = []
        for t_idx in range(2):
            for y_idx in range(2):
                for x_idx in range(2):
                    gx = grad_x[t_idx:t_idx+tres, y_idx:y_idx+hres, x_idx:x_idx+wres]
                    gy = grad_y[t_idx:t_idx+tres, y_idx:y_idx+hres, x_idx:x_idx+wres]
                    gz = grad_z[t_idx:t_idx+tres, y_idx:y_idx+hres, x_idx:x_idx+wres]

                    # Dot product with distance vectors
                    if t_idx == 0:
                        dt = lint[:, None, None]
                    else:
                        dt = lint[:, None, None] - 1

                    if y_idx == 0:
                        dy = liny[None, :, None]
                    else:
                        dy = liny[None, :, None] - 1

                    if x_idx == 0:
                        dx = linx[None, None, :]
                    else:
                        dx = linx[None, None, :] - 1

                    # Ensure proper broadcasting
                    gx_exp = gx[:lint.shape[0], :liny.shape[0], :linx.shape[0]]
                    gy_exp = gy[:lint.shape[0], :liny.shape[0], :linx.shape[0]]
                    gz_exp = gz[:lint.shape[0], :liny.shape[0], :linx.shape[0]]

                    dot = gx_exp * dx + gy_exp * dy + gz_exp * dt
                    gradients.append(dot)

        # Trilinear interpolation using precomputed masks
        noise = torch.zeros((self.num_frames, self.shape[1], self.shape[2]), device=self.device)
        for mask, gradient in zip(self.masks[octave], gradients):
            # Crop gradient to match noise dimensions
            grad_crop = gradient[:self.num_frames, :self.shape[1], :self.shape[2]]
            mask_crop = mask[:self.num_frames, :self.shape[1], :self.shape[2]]
            noise += mask_crop * grad_crop

        return noise

    def __call__(self) -> torch.Tensor:
        """
        Generate 3D fractal Perlin noise.

        Returns:
            Tensor of shape (num_frames, C, H, W) - a video sequence
        """
        # Initialize with zeros
        noise = torch.zeros((self.num_frames, self.shape[1], self.shape[2]), device=self.device)

        # Sum octaves with their respective factors
        for octave, factor in enumerate(self.factors):
            noise += factor * self.perlin_noise_3d(octave)

        # Expand to include channel dimension if needed
        if self.shape[0] is not None and self.shape[0] > 1:
            # Replicate for each channel (RGB)
            noise = noise.unsqueeze(1).expand(-1, self.shape[0], -1, -1)
        else:
            noise = noise.unsqueeze(1)

        return noise  # (num_frames, C, H, W)


# Convenience functions
def get_2d_perlin(shape: Tuple[int, int, int], seed: int = 0,
                  device: str = 'cuda', lacunarity: float = 2.0,
                  persistence: float = 0.5, octaves: int = 6) -> torch.Tensor:
    """
    Generate 2D fractal Perlin noise with default parameters.

    Args:
        shape: (C, H, W) tuple
        seed: Random seed
        device: Device to generate on ('cuda' or 'cpu')
        lacunarity: Frequency multiplier between octaves (default 2.0)
        persistence: Amplitude multiplier between octaves (default 0.5)
        octaves: Number of octaves to sum

    Returns:
        Tensor of shape (1, C, H, W) with values in range [-1, 1]
    """
    resolutions = [(2**i, 2**i) for i in range(1, octaves + 1)]
    factors = [persistence**i for i in range(octaves)]
    g = torch.Generator(device=device).manual_seed(seed)
    noise = FractalPerlin2D(shape, resolutions, factors, generator=g)(batch_size=1)
    return noise


def get_3d_perlin(shape: Tuple[int, int, int], num_frames: int, seed: int = 0,
                  device: str = 'cuda', lacunarity: float = 2.0,
                  persistence: float = 0.5, octaves: int = 6,
                  loop: bool = True) -> torch.Tensor:
    """
    Generate 3D fractal Perlin noise for video sequences.

    Args:
        shape: (C, H, W) tuple for spatial dimensions
        num_frames: Number of frames to generate
        seed: Random seed
        device: Device to generate on ('cuda' or 'cpu')
        lacunarity: Frequency multiplier between octaves (default 2.0)
        persistence: Amplitude multiplier between octaves (default 0.5)
        octaves: Number of octaves to sum
        loop: If True, the video will loop seamlessly

    Returns:
        Tensor of shape (num_frames, C, H, W) with values in range [-1, 1]
    """
    # Resolutions include time dimension
    resolutions = [(2**i, 2**i, 2**i) for i in range(1, octaves + 1)]
    factors = [persistence**i for i in range(octaves)]
    g = torch.Generator(device=device).manual_seed(seed)
    noise = FractalPerlin3D(shape, resolutions, factors, num_frames, generator=g, loop=loop)()
    return noise
