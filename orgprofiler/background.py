import numpy as np
from typing import Optional, Tuple
from scipy import ndimage as ndi
from functools import lru_cache
from skimage.morphology import disk

@lru_cache(maxsize=64)
def create_disk_structuring_element(radius: int) -> np.ndarray:
    """Create a cached boolean disk structuring element."""
    return disk(int(radius)).astype(bool, copy=False)

def estimate_background_from_ring(
    image_uint8: np.ndarray,
    foreground_mask: np.ndarray,
    ring_width_pixels: int = 20,
    method: str = "median",
    *,
    bounding_box: Optional[Tuple[int, int, int, int]] = None,
    padding: int = 2,
    algorithm: str = "edt",
    max_samples: int = 250_000,
) -> float:
    """
    Estimate background intensity from a ring around the foreground mask.
    """
    ring_radius = max(1, int(ring_width_pixels))
    image_height, image_width = image_uint8.shape[:2]

    # Determine bounding box for region of interest
    if bounding_box is None:
        foreground_rows, foreground_cols = np.nonzero(foreground_mask)
        if foreground_rows.size == 0:
            return 0.0
        min_row, max_row = int(foreground_rows.min()), int(foreground_rows.max())
        min_col, max_col = int(foreground_cols.min()), int(foreground_cols.max())
    else:
        min_row, min_col, max_row, max_col = map(int, bounding_box)

    # Expand bounding box
    min_row = max(0, min_row - ring_radius - padding)
    max_row = min(image_height, max_row + ring_radius + padding)
    min_col = max(0, min_col - ring_radius - padding)
    max_col = min(image_width, max_col + ring_radius + padding)

    roi_mask = foreground_mask[min_row:max_row, min_col:max_col].astype(bool, copy=False)
    roi_image = image_uint8[min_row:max_row, min_col:max_col]

    if algorithm == "edt":
        background_region = ~roi_mask
        if not background_region.any():
            background_region = ~foreground_mask
            roi_image = image_uint8

        distance_from_foreground = ndi.distance_transform_edt(background_region)
        ring_mask = (distance_from_foreground > 0) & (distance_from_foreground <= ring_radius)
    else:
        dilated_mask = ndi.binary_dilation(
            roi_mask,
            structure=create_disk_structuring_element(ring_radius),
            iterations=1
        )
        ring_mask = dilated_mask & ~roi_mask

    background_values = roi_image[ring_mask]
    if background_values.size == 0:
        background_values = roi_image[~roi_mask]
    if background_values.size == 0:
        return 0.0

    if background_values.size > max_samples:
        random_indices = np.random.randint(0, background_values.size, size=max_samples, dtype=np.int64)
        background_values = background_values[random_indices]

    return float(np.median(background_values)) if method == "median" else float(background_values.mean())

