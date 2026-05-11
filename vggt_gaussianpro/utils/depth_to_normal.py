"""
Compute surface normal maps from VGGT depth/point-map outputs.

All inputs/outputs are NumPy arrays; no GPU required at this stage.
"""

import numpy as np


def point_map_to_normals(points_3d: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel surface normals from a world-space 3-D point map.

    The normal at pixel (r, c) is estimated as the cross product of two
    finite-difference tangent vectors taken from the four-connected
    neighbourhood.  Edge rows/columns replicate their nearest neighbour.

    Args:
        points_3d: float32 array of shape (H, W, 3) — world-space XYZ per pixel.

    Returns:
        normals: float32 array of shape (H, W, 3) — unit surface normals.
            Pixels where the cross product is degenerate (e.g. depth == 0)
            are assigned the up-vector [0, 0, 1].
    """
    H, W, _ = points_3d.shape

    # Horizontal tangent: right neighbour minus left neighbour (central diff).
    dX = np.empty_like(points_3d)
    dX[:, 1:-1] = points_3d[:, 2:] - points_3d[:, :-2]
    dX[:, 0]    = points_3d[:, 1]  - points_3d[:, 0]
    dX[:, -1]   = points_3d[:, -1] - points_3d[:, -2]

    # Vertical tangent: bottom minus top.
    dY = np.empty_like(points_3d)
    dY[1:-1, :] = points_3d[2:, :] - points_3d[:-2, :]
    dY[0, :]    = points_3d[1, :]  - points_3d[0, :]
    dY[-1, :]   = points_3d[-1, :] - points_3d[-2, :]

    normals = np.cross(dX, dY)  # (H, W, 3)

    norms = np.linalg.norm(normals, axis=-1, keepdims=True)  # (H, W, 1)
    degenerate = (norms < 1e-8).squeeze(-1)

    norms = np.where(norms < 1e-8, 1.0, norms)
    normals = normals / norms

    # Assign a safe default for degenerate pixels.
    normals[degenerate] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    return normals.astype(np.float32)


def batch_point_map_to_normals(points_3d_batch: np.ndarray) -> np.ndarray:
    """
    Vectorised version for a batch of point maps.

    Args:
        points_3d_batch: float32 array of shape (N, H, W, 3).

    Returns:
        normals: float32 array of shape (N, H, W, 3).
    """
    return np.stack(
        [point_map_to_normals(points_3d_batch[i]) for i in range(len(points_3d_batch))],
        axis=0,
    )
