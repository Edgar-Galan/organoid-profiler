import numpy as np
from typing import Optional, Union, Tuple
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation
try:
    from skimage.morphology import footprint_rectangle as rectangle
except ImportError:
    from skimage.morphology import rectangle
from skimage.morphology import disk
from skimage.segmentation import clear_border
from pathlib import Path
from loguru import logger

from .imaging import convert_rgb_to_grayscale_uint8

try:
    from cellpose import models, core
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False

DEFAULT_STRUCTURING_ELEMENT_SIZE = (3, 3)
MODELS_DIR = Path(__file__).parent.parent / "models" 

def resolve_model_path(name: str) -> str:
    """Look up model in repo/models/models/."""
    path = MODELS_DIR / name
    return str(path.resolve()) if path.exists() else name

def build_segmentation_mask_fiji_style(
    image_rgb: np.ndarray,
    *,
    gaussian_sigma: float,
    dilation_iterations: int,
    erosion_iterations: int,
    clear_border_artifacts: bool = True,
    object_is_dark: bool = True,
) -> np.ndarray:
    """Build binary mask using a Fiji/ImageJ-inspired pipeline."""
    grayscale = convert_rgb_to_grayscale_uint8(image_rgb)
    initial_threshold = filters.threshold_isodata(grayscale)
    
    binary_mask = (grayscale <= initial_threshold) if object_is_dark else (grayscale >= initial_threshold)
    binary_mask = ndi.binary_fill_holes(binary_mask)

    structuring_element = rectangle(DEFAULT_STRUCTURING_ELEMENT_SIZE)

    for _ in range(int(dilation_iterations)):
        binary_mask = morphology.binary_dilation(binary_mask, footprint=structuring_element)
    binary_mask = ndi.binary_fill_holes(binary_mask)

    for _ in range(int(erosion_iterations)):
        binary_mask = morphology.binary_erosion(binary_mask, footprint=structuring_element)
    binary_mask = ndi.binary_fill_holes(binary_mask)

    mask_float32 = binary_mask.astype(np.float32, copy=False)
    smoothed_mask_float32 = np.empty_like(mask_float32, dtype=np.float32)
    ndi.gaussian_filter(mask_float32, sigma=gaussian_sigma, output=smoothed_mask_float32, mode="nearest")

    smoothed_mask_uint8 = (smoothed_mask_float32 * 255).astype(np.uint8)
    secondary_threshold_uint8 = filters.threshold_isodata(smoothed_mask_uint8)
    final_mask = (smoothed_mask_float32 >= (secondary_threshold_uint8 / 255.0))

    if clear_border_artifacts:
        final_mask = clear_border(final_mask)
    if final_mask.mean() > 0.8:
        final_mask = ~final_mask

    return final_mask

def build_segmentation_mask_cpsam(
    image_rgb: np.ndarray,
    *,
    diameter: Optional[float] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_size: int = 15,
    clear_border_artifacts: bool = True,
    return_flows: bool = False,
    model_type: str = "cpsam",
    pretrained_model: Optional[str] = "organoid_cpsam",
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Build binary mask using Cellpose model (supports cpsam, and custom models)."""
    if not CELLPOSE_AVAILABLE:
        raise RuntimeError("cellpose is not installed")

    grayscale = convert_rgb_to_grayscale_uint8(image_rgb)
    
    # In Cellpose 4, we use Cellpose for built-in models and CellposeModel for custom ones
    if pretrained_model:
        model = models.CellposeModel(pretrained_model=resolve_model_path(pretrained_model), gpu=core.use_gpu())
        logger.info(f"Evaluating organoid cpsam...")
    else:
        # Use Cellpose class for built-in models to avoid v4 warnings
        model = models.Cellpose(model_type=model_type, gpu=core.use_gpu())
        logger.info(f"Evaluating cpsam")
    # Cellpose 4 eval signature is mostly backward compatible, but supports cpsam
    # Ensure diameter is not 0 to avoid ZeroDivisionError
    eval_diameter = diameter if diameter and diameter > 0 else None
    
    result = model.eval(
        grayscale.astype(np.uint8, copy=False),
        diameter=eval_diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
        channels=[0, 0],
    )
    
    masks = result[0] if isinstance(result, tuple) else result
    flows = result[1] if isinstance(result, tuple) else None
    
    binary_mask = (masks > 0).astype(bool, copy=False)

    if clear_border_artifacts:
        binary_mask = clear_border(binary_mask)

    if return_flows and flows is not None:
        # flows[0] is the flow RGB image from Cellpose
        return binary_mask, flows[0]
    
    return binary_mask

