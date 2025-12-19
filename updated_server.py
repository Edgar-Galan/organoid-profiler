# main.py
from __future__ import annotations
import datetime
import json
from typing import Tuple, Optional, Literal, Union, List
import contextlib
from functools import lru_cache

import base64
import io
import os
import math
import re
import sys
from typing import Any, Dict
from uuid import uuid4
import os, time, platform

import anyio
import httpx
import numpy as np
from PIL import Image
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from skimage import filters, measure, morphology, segmentation
from skimage.morphology import rectangle, disk
from skimage.measure import perimeter_crofton
from skimage.segmentation import clear_border
from supabase import Client, create_client
from loguru import logger

# Cellpose (optional import)
try:
    from cellpose import models, core
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    logger.warning("cellpose not installed; cyto2 segmentation disabled")


# ----------------------------
# Supabase client (database)
# ----------------------------

class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_KEY: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False, 
    )

settings = Settings()

url = settings.SUPABASE_URL
key = settings.SUPABASE_KEY
supabase: Client = create_client(url, key)

# ----------------------------
# Image Processing Constants
# ----------------------------

# Morphological structuring element size
DEFAULT_STRUCTURING_ELEMENT_SIZE = (3, 3)


# ----------------------------
# Small utils
# ----------------------------

@lru_cache(maxsize=64)
def create_disk_structuring_element(radius: int) -> np.ndarray:
    """Create a cached boolean disk structuring element for morphological operations."""
    return disk(int(radius)).astype(bool, copy=False)

# ----------------------------
# Logging (Loguru)
# ----------------------------
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    enqueue=True,
    backtrace=True,
    diagnose=True,
    filter=lambda record: record["level"].name == "INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
           "<level>{level: <7}</level> | "
           "{name}:{function}:{line} - <level>" \
           "{message}</level>",
)
logger.add(
    sys.stderr,
    level="DEBUG",  
    enqueue=True,
    backtrace=True,
    diagnose=True,
    filter=lambda record: record["level"].name == "DEBUG",
    format="<blue>{time:YYYY-MM-DD HH:mm:ss.SSS}</blue> | "
           "<level>{level: <7}</level> | "
           "{name}:{function}:{line} - <level>{message}</level>",
)


# ----------------------------
# psutil + peak RSS helpers
# ----------------------------
try:
    import psutil
except Exception:
    psutil = None
    logger.warning("psutil not installed; resource profiling disabled")

def get_max_resident_set_size_bytes() -> Optional[int]:
    """Get the maximum resident set size (peak memory usage) in bytes."""
    try:
        import resource
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform.startswith("linux"):
            return int(max_rss) * 1024
        return int(max_rss)
    except Exception:
        return None

@contextlib.contextmanager
def time_block(label: str):
    _t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - _t0
        logger.info(f"[TIMER] {label}: {dt:.3f}s")


@contextlib.contextmanager
def timed(timings: Dict[str, float], key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = round(time.perf_counter() - t0, 6)

class ResourceProfiler:
    def __init__(self, label: str = "analyze"):
        self.label = label
        self.metrics: Dict[str, float] = {}

    def __enter__(self):
        self.t0 = time.perf_counter()
        if psutil:
            self.proc = psutil.Process(os.getpid())
            self.ct0 = self.proc.cpu_times()
            self.mem0 = self.proc.memory_info().rss
        else:
            self.proc = None
            self.ct0 = None
            self.mem0 = None
        self.maxrss0 = get_max_resident_set_size_bytes()
        return self

    def __exit__(self, exc_type, exc, tb):
        t1 = time.perf_counter()
        wall_s = t1 - self.t0

        cpu_user_s = cpu_sys_s = rss_now = rss_delta = None
        if self.proc:
            ct1 = self.proc.cpu_times()
            mi1 = self.proc.memory_info()
            cpu_user_s = (ct1.user - self.ct0.user)
            cpu_sys_s  = (ct1.system - self.ct0.system)
            rss_now    = mi1.rss
            rss_delta  = (rss_now - self.mem0)

        maxrss1 = get_max_resident_set_size_bytes()
        peak_rss_bytes = None
        if maxrss1 is not None and self.maxrss0 is not None:
            peak_rss_bytes = max(0, maxrss1 - self.maxrss0) or maxrss1

        def bytes_to_mb(bytes_value):
            return None if bytes_value is None else round(bytes_value / (1024*1024), 3)

        self.metrics = {
            "wall_time_s": round(wall_s, 6),
            "cpu_user_s": None if cpu_user_s is None else round(cpu_user_s, 6),
            "cpu_sys_s": None if cpu_sys_s is None else round(cpu_sys_s, 6),
            "rss_now_bytes": rss_now,
            "rss_now_mb": bytes_to_mb(rss_now),
            "rss_delta_bytes": rss_delta,
            "rss_delta_mb": bytes_to_mb(rss_delta),
            "peak_rss_bytes": peak_rss_bytes,
            "peak_rss_mb": bytes_to_mb(peak_rss_bytes),
            "platform": platform.platform(),
        }

        logger.info(
            f"[{self.label}] wall={wall_s:.3f}s "
            f"cpu_user={self.metrics['cpu_user_s']}s cpu_sys={self.metrics['cpu_sys_s']}s "
            f"rss_now={self.metrics['rss_now_mb']}MB Δrss={self.metrics['rss_delta_mb']}MB "
            f"peak_rss={self.metrics['peak_rss_mb']}MB"
        )

# ----------------------------
# FastAPI setup
# ----------------------------
api = FastAPI()

# Add exception handler for unhandled exceptions (but not HTTPException)
@api.exception_handler(Exception)
async def global_exception_handler(request, exc):
    if isinstance(exc, HTTPException):
        raise  # Re-raise HTTPExceptions
    logger.exception(f"Unhandled exception in {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@api.get("/healthz")
def healthz():
    return {"ok": True}

@api.get("/healthz/db")
def healthz_db():
    """Check if Supabase database connection is working."""
    try:
        # Try a simple query to verify connection
        result = supabase.table("analysis_runs").select("id").limit(1).execute()
        return {
            "ok": True,
            "database": "connected",
            "tables_accessible": True,
        }
    except Exception as e:
        logger.exception("Database health check failed")
        raise HTTPException(503, f"Database connection failed: {str(e)}")

# ----------------------------
# Helpers
# ----------------------------
def convert_array_to_data_url_png(array: np.ndarray) -> str:
    """Convert a numpy array to a base64-encoded PNG data URL."""
    converted_array = array
    if converted_array.dtype != np.uint8:
        converted_array = np.clip(converted_array, 0, 255).astype(np.uint8)

    mode = "L" if converted_array.ndim == 2 else "RGB"
    image = Image.fromarray(converted_array if converted_array.ndim == 2 else converted_array, mode=mode)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return f"data:image/png;base64,{base64_encoded}"

def convert_rgb_to_grayscale_uint8(image_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale using standard luminance weights (ITU-R BT.601)."""
    red_channel = image_rgb[..., 0].astype(np.float32)
    green_channel = image_rgb[..., 1].astype(np.float32)
    blue_channel = image_rgb[..., 2].astype(np.float32)

    # Standard RGB to grayscale conversion weights
    grayscale = 0.299 * red_channel + 0.587 * green_channel + 0.114 * blue_channel
    return np.clip(grayscale, 0, 255).astype(np.uint8)

def calculate_feret_features_from_points(points_xy: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Calculate Feret diameter features from a set of points.

    Returns:
        max_feret: Maximum Feret diameter
        min_feret: Minimum Feret diameter (caliper width)
        feret_angle: Angle of maximum Feret diameter in degrees
        feret_x: X-coordinate of Feret diameter start point
        feret_y: Y-coordinate of Feret diameter start point
    """
    if points_xy.shape[0] < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Find convex hull of the points
    convex_hull = ConvexHull(points_xy)
    hull_points = points_xy[convex_hull.vertices]

    # Find maximum distance between hull points (max Feret diameter)
    max_distance_squared = 0.0
    best_start_index = 0
    best_end_index = 0

    for i in range(len(hull_points)):
        for j in range(i + 1, len(hull_points)):
            delta_x = hull_points[j, 0] - hull_points[i, 0]
            delta_y = hull_points[j, 1] - hull_points[i, 1]
            distance_squared = delta_x * delta_x + delta_y * delta_y

            if distance_squared > max_distance_squared:
                max_distance_squared = distance_squared
                best_start_index = i
                best_end_index = j

    max_feret = math.sqrt(max_distance_squared)
    feret_angle = math.degrees(math.atan2(
        hull_points[best_end_index, 1] - hull_points[best_start_index, 1],
        hull_points[best_end_index, 0] - hull_points[best_start_index, 0]
    ))
    feret_x = float(hull_points[best_start_index, 0])
    feret_y = float(hull_points[best_start_index, 1])

    def calculate_width_for_edge(point_0, point_1):
        """Calculate the width perpendicular to an edge."""
        edge_vector_x = point_1[0] - point_0[0]
        edge_vector_y = point_1[1] - point_0[1]
        edge_length = math.hypot(float(edge_vector_x), float(edge_vector_y))

        if edge_length == 0:
            return float("inf")

        # Normal vector (perpendicular to edge)
        normal_x = -edge_vector_y / edge_length
        normal_y = edge_vector_x / edge_length

        # Project all hull points onto the normal
        projections = hull_points @ np.array([normal_x, normal_y], dtype=float)
        return float(projections.max() - projections.min())

    # Find minimum width (caliper width)
    min_feret = float("inf")
    for i in range(len(hull_points)):
        width = calculate_width_for_edge(hull_points[i], hull_points[(i + 1) % len(hull_points)])
        if width < min_feret:
            min_feret = width

    return float(max_feret), float(min_feret), float(feret_angle), feret_x, feret_y

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

    Args:
        image_uint8: Input grayscale image (uint8)
        foreground_mask: Binary mask of foreground region
        ring_width_pixels: Width of the background ring in pixels
        method: "median" or "mean" for aggregating background values
        bounding_box: Optional (min_row, min_col, max_row, max_col) to limit search region
        padding: Extra padding around bounding box
        algorithm: "edt" for Euclidean distance transform or "dilation" for morphological dilation
        max_samples: Maximum number of pixels to sample from the ring

    Returns:
        Estimated background intensity value
    """
    ring_radius = max(1, int(ring_width_pixels))
    image_height, image_width = image_uint8.shape[:2]

    # Determine bounding box for region of interest
    if bounding_box is None:
        foreground_rows, foreground_cols = np.nonzero(foreground_mask)
        if foreground_rows.size == 0:
            return 0.0
        min_row = int(foreground_rows.min())
        max_row = int(foreground_rows.max())
        min_col = int(foreground_cols.min())
        max_col = int(foreground_cols.max())
    else:
        min_row, min_col, max_row, max_col = map(int, bounding_box)

    # Expand bounding box with padding
    min_row = max(0, min_row - ring_radius - padding)
    max_row = min(image_height, max_row + ring_radius + padding)
    min_col = max(0, min_col - ring_radius - padding)
    max_col = min(image_width, max_col + ring_radius + padding)

    # Extract region of interest
    roi_mask = foreground_mask[min_row:max_row, min_col:max_col].astype(bool, copy=False)
    roi_image = image_uint8[min_row:max_row, min_col:max_col]

    # Create ring region around foreground
    if algorithm == "edt":
        # Use Euclidean distance transform
        background_region = ~roi_mask
        if not background_region.any():
            background_region = ~foreground_mask
            roi_image = image_uint8

        distance_from_foreground = ndi.distance_transform_edt(background_region)
        ring_mask = (distance_from_foreground > 0) & (distance_from_foreground <= ring_radius)
    else:
        # Use morphological dilation
        dilated_mask = ndi.binary_dilation(
            roi_mask,
            structure=create_disk_structuring_element(ring_radius),
            iterations=1
        )
        ring_mask = dilated_mask & ~roi_mask

    # Extract background pixel values from ring
    background_values = roi_image[ring_mask]
    if background_values.size == 0:
        # Fallback: use all background pixels
        background_values = roi_image[~roi_mask]
    if background_values.size == 0:
        return 0.0

    # Subsample if too many pixels
    if background_values.size > max_samples:
        random_indices = np.random.randint(0, background_values.size, size=max_samples, dtype=np.int64)
        background_values = background_values[random_indices]

    # Return median or mean
    if method == "median":
        return float(np.median(background_values))
    else:
        return float(background_values.mean())

# ----------------------------
# Mask builder (Fiji-like path)
# ----------------------------
def build_segmentation_mask_fiji_style(
    image_rgb: np.ndarray,
    *,
    gaussian_sigma: float,
    dilation_iterations: int,
    erosion_iterations: int,
    clear_border_artifacts: bool = True,
    object_is_dark: bool = True,
) -> np.ndarray:
    """
    Build a binary segmentation mask using a Fiji/ImageJ-inspired pipeline.

    This implements a multi-stage thresholding and morphological processing approach:
    1. Initial thresholding using Isodata algorithm
    2. Morphological operations (dilation/erosion) with hole filling
    3. Gaussian smoothing
    4. Secondary thresholding on the smoothed result

    Args:
        image_rgb: Input RGB image
        gaussian_sigma: Standard deviation for Gaussian blur
        dilation_iterations: Number of dilation iterations
        erosion_iterations: Number of erosion iterations
        clear_border_artifacts: Whether to clear objects touching the border
        object_is_dark: True if objects are darker than background, False otherwise

    Returns:
        Binary mask where True indicates foreground (cells/objects)
    """
    # Convert to grayscale
    grayscale = convert_rgb_to_grayscale_uint8(image_rgb)

    # Initial thresholding
    initial_threshold = filters.threshold_isodata(grayscale)
    if object_is_dark:
        binary_mask = (grayscale <= initial_threshold)
    else:
        binary_mask = (grayscale >= initial_threshold)

    binary_mask = ndi.binary_fill_holes(binary_mask)

    # Morphological operations
    structuring_element = rectangle(DEFAULT_STRUCTURING_ELEMENT_SIZE)

    # Dilation phase
    for _ in range(int(dilation_iterations)):
        binary_mask = morphology.binary_dilation(binary_mask, footprint=structuring_element)
    binary_mask = ndi.binary_fill_holes(binary_mask)

    # Erosion phase
    for _ in range(int(erosion_iterations)):
        binary_mask = morphology.binary_erosion(binary_mask, footprint=structuring_element)
    binary_mask = ndi.binary_fill_holes(binary_mask)

    # Gaussian smoothing
    mask_float32 = binary_mask.astype(np.float32, copy=False)
    smoothed_mask_float32 = np.empty_like(mask_float32, dtype=np.float32)
    ndi.gaussian_filter(mask_float32, sigma=gaussian_sigma, output=smoothed_mask_float32, mode="nearest")

    # Secondary thresholding on smoothed mask
    smoothed_mask_uint8 = (smoothed_mask_float32 * 255).astype(np.uint8)
    secondary_threshold_uint8 = filters.threshold_isodata(smoothed_mask_uint8)
    threshold_normalized = secondary_threshold_uint8 / 255.0

    final_mask = (smoothed_mask_float32 >= threshold_normalized)

    # Clean up border artifacts
    if clear_border_artifacts:
        final_mask = clear_border(final_mask)

    # Invert if most of the image is foreground (likely inverted)
    if final_mask.mean() > 0.8:
        final_mask = ~final_mask

    return final_mask


# ----------------------------
# Mask builder (Cellpose cyto2 path)
# ----------------------------
@lru_cache(maxsize=1)
def _get_cyto2_model():
    """Get or create a cached cyto2 Cellpose model instance."""
    if not CELLPOSE_AVAILABLE:
        raise ImportError("cellpose is not installed")
    return models.Cellpose(model_type="cyto2", gpu=core.use_gpu())

def build_segmentation_mask_cyto2(
    image_rgb: np.ndarray,
    *,
    diameter: Optional[float] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_size: int = 15,
    clear_border_artifacts: bool = True,
) -> np.ndarray:
    """
    Build a binary segmentation mask using Cellpose cyto2 model.

    This uses the pre-trained cyto2 model from Cellpose to segment cells/objects.
    The model is cached after first use for performance.

    Args:
        image_rgb: Input RGB image (uint8, shape HxWx3)
        diameter: Expected cell diameter in pixels. If None, Cellpose will estimate.
        flow_threshold: Flow error threshold (lower = more strict, default 0.4)
        cellprob_threshold: Cell probability threshold (default 0.0, lower = more cells)
        min_size: Minimum size of objects to keep (in pixels, default 15)
        clear_border_artifacts: Whether to clear objects touching the border

    Returns:
        Binary mask where True indicates foreground (cells/objects)
    """
    if not CELLPOSE_AVAILABLE:
        raise RuntimeError("cellpose is not installed. Install with: pip install cellpose")

    logger.info(
        f"Cellpose cyto2 segmentation | diameter={diameter} "
        f"flow_threshold={flow_threshold} cellprob_threshold={cellprob_threshold}"
    )

    # Convert RGB to grayscale for Cellpose (expects 2D or 3D grayscale)
    grayscale = convert_rgb_to_grayscale_uint8(image_rgb)

    # Cellpose expects images in specific format
    # For 2D grayscale, ensure it's uint8
    image_for_cellpose = grayscale.astype(np.uint8, copy=False)

    # Get cached model
    model = _get_cyto2_model()

    # Run Cellpose segmentation
    # Cellpose returns masks (labeled image), flows, styles, and estimated diameter
    result = model.eval(
        image_for_cellpose,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
        channels=[0, 0],  # [grayscale, grayscale] for single-channel input
    )
    
    # Handle return values (Cellpose API may vary by version)
    if isinstance(result, tuple):
        masks = result[0]
        diams = result[3] if len(result) > 3 else diameter
    else:
        masks = result
        diams = diameter

    # Convert labeled mask to binary mask (any label > 0 is foreground)
    binary_mask = (masks > 0).astype(bool, copy=False)

    # Clean up border artifacts if requested
    if clear_border_artifacts:
        binary_mask = clear_border(binary_mask)

    # Log statistics
    num_objects = int(masks.max()) if masks.size > 0 else 0
    foreground_ratio = float(binary_mask.mean())
    diams_str = f"{diams:.1f}" if diams is not None else "auto"
    logger.info(
        f"Cellpose cyto2 done | objects={num_objects} "
        f"foreground_ratio={foreground_ratio:.3f} estimated_diameter={diams_str}px"
    )

    return binary_mask


# ----------------------------
# Core analysis (supports both BF & FL via args)
# ----------------------------
def analyze_image(
    img: np.ndarray,
    *,
    sigma_pre: float,
    dilate_iter: int,
    erode_iter: int,
    min_area_px: float,
    max_area_px: float,
    min_circ: float,
    edge_margin: float,
    pixel_size_um: float,
    overlay_width: int,
    return_images: bool,
    crop_overlay: bool,
    crop_border_px: int,
    ring_px: int,
    # differences between BF and FL:
    invert_for_intensity: bool,
    exclude_edge_particles: bool,
    select_strategy: Literal["largest", "composite_filtered"],
    area_filter_px: Optional[float],
    background_mode: Literal["ring", "inverse_of_composite"],
    object_is_dark: bool,
    # ---- NEW: fluorescence-only behavior toggle ----
    ignore_zero_bins_for_mode_min: bool = False,
) -> Dict[str, Any]:

    H, W, _ = img.shape
    logger.info(
        f"Analyze start | {H}x{W} σ={sigma_pre} dilate={dilate_iter} erode={erode_iter} "
        f"minArea={min_area_px} minCirc={min_circ} edgeMargin={edge_margin}"
    )

    # Optional border crop (both modes)
    with time_block("optional border crop"):
        if crop_border_px > 0 and min(img.shape[:2]) > 2 * crop_border_px:
            img = img[crop_border_px:-crop_border_px, crop_border_px:-crop_border_px, :]
            H, W, _ = img.shape

    # Build mask (common)
    with time_block("build_segmentation_mask_fiji_style"):
        mask_bool = build_segmentation_mask_fiji_style(
            image_rgb=img,
            gaussian_sigma=sigma_pre,
            dilation_iterations=dilate_iter,
            erosion_iterations=erode_iter,
            object_is_dark=object_is_dark,
        )

    with time_block("label mask + regionprops (initial)"):
        labels = measure.label(mask_bool, connectivity=2)
        if labels.max() == 0:
            logger.warning("No contours found after masking")
            raise HTTPException(422, "No contours found")

    xmin, xmax = W * edge_margin, W * (1.0 - edge_margin)
    ymin, ymax = H * edge_margin, H * (1.0 - edge_margin)

    # Filter components (area, circularity, centroid-in-margin if requested)
    with time_block("filter components"):
        keep_mask = np.zeros_like(labels, dtype=bool)
        for r in measure.regionprops(labels):
            a_px = float(r.area)
            if a_px < min_area_px or a_px > max_area_px:
                continue
            p_px = float(perimeter_crofton(labels == r.label, directions=4))
            circ_f = (4.0 * math.pi * a_px) / (p_px * p_px) if p_px > 0 else 0.0
            if circ_f < min_circ:
                continue
            if exclude_edge_particles:
                cy, cx = r.centroid
                if not (xmin <= cx <= xmax and ymin <= cy <= ymax):
                    continue
            keep_mask |= (labels == r.label)

        # Fluorescence: drop small ROIs after first pass (composite filter)
        if area_filter_px is not None and keep_mask.any():
            lab2 = measure.label(keep_mask, connectivity=2)
            keep2 = np.zeros_like(keep_mask, dtype=bool)
            for r in measure.regionprops(lab2):
                if float(r.area) >= float(area_filter_px):
                    keep2 |= (lab2 == r.label)
            keep_mask = keep2

    with time_block("fallback to largest (if needed)"):
        if not keep_mask.any():
            largest = max(measure.regionprops(labels), key=lambda rr: rr.area)
            logger.info("Filters removed all; falling back to largest region")
            keep_mask = (labels == largest.label)

    # Selection strategy
    with time_block("regionprops (final) + measurements"):
        if select_strategy == "largest":
            union_lab = measure.label(keep_mask, connectivity=2)
            regs = measure.regionprops(union_lab)
            if not regs:
                raise HTTPException(422, "Empty ROI after masking")
            props = max(regs, key=lambda r: r.area)
            mask_measured = (union_lab == props.label)
            major_px = float(props.major_axis_length or 0.0)
            minor_px = float(props.minor_axis_length or 0.0)
            angle = float(np.degrees(props.orientation or 0.0))
            solidity = float(getattr(props, "solidity", 0.0))
            minr, minc, maxr, maxc = props.bbox
            cy_px, cx_px = props.centroid
        else:
            # composite union
            mask_measured = keep_mask
            ys, xs = np.nonzero(mask_measured)
            if ys.size == 0:
                raise HTTPException(422, "Empty ROI after masking")

            # Compute regionprops on the union-as-one-label
            union_lbl = mask_measured.astype(np.uint8)  # all foreground = label 1
            rp = measure.regionprops(union_lbl)[0]

            minr, minc, maxr, maxc = rp.bbox
            cy_px, cx_px = rp.centroid
            major_px = float(rp.major_axis_length or 0.0)
            minor_px = float(rp.minor_axis_length or 0.0)
            angle    = float(np.degrees(rp.orientation or 0.0))
            solidity = float(getattr(rp, "solidity", 0.0))

        area_px = float(mask_measured.sum())
        perim_px = float(perimeter_crofton(mask_measured, directions=4))

    # Feret from boundary points (works for both strategies)
    with time_block("feret from contours"):
        contours = measure.find_contours(mask_measured, 0.5)
        if contours:
            cont = max(contours, key=lambda c: c.shape[0])
            if cont.shape[0] > 4000:
                cont = cont[::4]
            pts = np.column_stack((cont[:, 1], cont[:, 0]))  # (x, y)
        else:
            pts = np.empty((0, 2))

        if pts.shape[0] >= 3:
            feret_px, minFeret_px, feretAngle, feretX_px, feretY_px = calculate_feret_features_from_points(pts)
        else:
            feret_px = minFeret_px = feretAngle = feretX_px = feretY_px = 0.0

    # Convert to µm / µm²
    px_um = float(pixel_size_um)
    area = area_px * (px_um ** 2)
    perim = perim_px * px_um
    cx, cy = cx_px * px_um, cy_px * px_um
    major, minor = major_px * px_um, minor_px * px_um
    feret, minFeret = feret_px * px_um, minFeret_px * px_um
    feretX, feretY = feretX_px * px_um, feretY_px * px_um
    bx_px, by_px = float(minc), float(minr)
    width_px, height_px = float(maxc - minc), float(maxr - minr)
    bx, by = bx_px * px_um, by_px * px_um
    width, height = width_px * px_um, height_px * px_um

    circ = (4.0 * math.pi * area) / (perim * perim) if perim > 0 else 0.0
    ar = float(major / minor) if minor > 0 else 0.0
    roundness = float((4.0 * area) / (math.pi * major * major)) if major > 0 else 0.0

    # Intensity stats
    with time_block("intensity stats + COM"):
        grayscale_uint8 = convert_rgb_to_grayscale_uint8(img)
        intensity_image = (255 - grayscale_uint8).astype(np.uint8) if invert_for_intensity else grayscale_uint8
        intensity_values = intensity_image[mask_measured].astype(np.uint8)
        if intensity_values.size == 0:
            raise HTTPException(422, "Empty ROI (no intensity)")

        mean_intensity = float(intensity_values.mean())
        median_intensity = float(np.median(intensity_values))

        # ---- fluorescence-only handling of mode/min ----
        histogram = np.bincount(intensity_values, minlength=256)
        if ignore_zero_bins_for_mode_min and histogram[1:].sum() > 0:
            mode_intensity = float(histogram[1:].argmax() + 1)  # ignore 0-bin
            min_intensity_positive = float(intensity_values[intensity_values > 0].min())
        else:
            mode_intensity = float(np.argmax(histogram))
            min_intensity_positive = float(intensity_values.min())

        min_intensity = float(intensity_values.min())
        max_intensity = float(intensity_values.max())

        std_dev_intensity = float(intensity_values.std(ddof=0))
        if std_dev_intensity > 0:
            z_scores = (intensity_values.astype(np.float32) - mean_intensity) / std_dev_intensity
            skewness = float(np.mean(z_scores ** 3))
            kurtosis = float(np.mean(z_scores ** 4) - 3.0)
        else:
            skewness = kurtosis = 0.0

        raw_integrated_density = float(intensity_values.sum())
        integrated_density = raw_integrated_density
        center_of_mass_y_px, center_of_mass_x_px = ndi.center_of_mass(
            intensity_image,
            labels=mask_measured.astype(np.uint8),
            index=1
        )
        center_of_mass_x = float(center_of_mass_x_px * px_um)
        center_of_mass_y = float(center_of_mass_y_px * px_um)

    # Background + corrected metrics
    with time_block("background + corrected metrics"):
        if background_mode == "ring":
            background_intensity = estimate_background_from_ring(
                intensity_image,
                mask_measured,
                ring_width_pixels=ring_px,
                method="median",
                bounding_box=(int(by_px), int(bx_px), int(by_px + height_px), int(bx_px + width_px))
            )
        else:
            roi_intensity = intensity_image[int(by_px):int(by_px+height_px), int(bx_px):int(bx_px+width_px)]
            roi_background_mask = ~mask_measured[int(by_px):int(by_px+height_px), int(bx_px):int(bx_px+width_px)]
            background_values = roi_intensity[roi_background_mask]
            background_intensity = float(np.median(background_values)) if background_values.size else 0.0

        corrected_total_intensity = raw_integrated_density - background_intensity * area_px
        corrected_mean_intensity = mean_intensity - background_intensity
        # use positive-min only when the fluorescence flag is on
        corrected_min_intensity = (min_intensity_positive if ignore_zero_bins_for_mode_min else min_intensity) - background_intensity
        corrected_max_intensity = max_intensity - background_intensity
        equivalent_diameter = math.sqrt(4.0 * area / math.pi)
        centroid_to_center_of_mass_distance = math.hypot(
            float(center_of_mass_x - cx),
            float(center_of_mass_y - cy)
        )

    # Overlays
    with time_block("build overlays" if return_images else "skip overlays"):
        if return_images:
            if crop_overlay:
                cropped_mask = mask_measured[int(by_px):int(by_px+height_px), int(bx_px):int(bx_px+width_px)]
                overlay_image = img[int(by_px):int(by_px+height_px), int(bx_px):int(bx_px+width_px)].copy()
                boundary_pixels = segmentation.find_boundaries(cropped_mask, mode="outer")
            else:
                cropped_mask = mask_measured
                overlay_image = img.copy()
                boundary_pixels = segmentation.find_boundaries(cropped_mask, mode="outer")

            # Dilate boundaries for visibility
            boundary_radius = max(1, overlay_width // 2)
            dilated_boundaries = morphology.binary_dilation(boundary_pixels, disk(boundary_radius))

            # Draw magenta boundaries
            overlay_image[dilated_boundaries] = np.array([255, 0, 255], dtype=np.uint8)

            # Create mask visualization
            mask_visualization = (cropped_mask.astype(np.uint8) * 255)

            roi_image_base64 = convert_array_to_data_url_png(overlay_image)
            mask_image_base64 = convert_array_to_data_url_png(mask_visualization)
        else:
            roi_image_base64 = ""
            mask_image_base64 = ""

    results = {
        "area": area,
        "mean": mean_intensity,
        "stdDev": std_dev_intensity,
        "mode": mode_intensity,
        "min": min_intensity,
        "max": max_intensity,
        "x": cx,
        "y": cy,
        "xm": center_of_mass_x,
        "ym": center_of_mass_y,
        "perim": perim,
        "bx": bx,
        "by": by,
        "width": width,
        "height": height,
        "major": major,
        "minor": minor,
        "angle": angle,
        "circ": circ,
        "feret": feret,
        "intDen": integrated_density,
        "median": median_intensity,
        "skew": -skewness,  # Negated for ImageJ compatibility
        "kurt": kurtosis,
        "rawIntDen": raw_integrated_density,
        "feretX": feretX,
        "feretY": feretY,
        "feretAngle": feretAngle,
        "minFeret": minFeret,
        "ar": ar,
        "round": roundness,
        "solidity": solidity,
        "eqDiam": equivalent_diameter,
        "corrTotalInt": corrected_total_intensity,
        "corrMeanInt": corrected_mean_intensity,
        "corrMinInt": corrected_min_intensity,
        "corrMaxInt": corrected_max_intensity,
        "centroidToCom": centroid_to_center_of_mass_distance,
        "bgRing": background_intensity,
    }

    logger.info(
        f"Analyze done | area_px={area_px:.0f} perim_px={perim_px:.1f} "
        f"circ={circ:.3f} eq_diam={equivalent_diameter:.2f}µm"
    )
    return {
        "results": results,
        "roi_image": roi_image_base64,
        "mask_image": mask_image_base64,
    }

def _result_row_from_payload(run_id: str, filename: str, payload: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
    r = payload["results"]
    return {
        "run_id": run_id,
        "filename": filename,
        "analysis_type": analysis_type,
        # Storage links (filled after upload)
        "roi_image_path": None,
        "mask_image_path": None,

        # parsed ids / growth
        "day": r.get("day"),
        "organoid_number": r.get("organoidNumber"),
        "growth_rate": r.get("growthRate"),

        # metrics (map 1:1 with your table)
        "area": r.get("area"),
        "perim": r.get("perim"),
        "eqDiam": r.get("eqDiam"),
        "circ": r.get("circ"),
        "ar": r.get("ar"),
        "roundness": r.get("round"),
        "solidity": r.get("solidity"),
        "major": r.get("major"),
        "minor": r.get("minor"),
        "angle": r.get("angle"),
        "feret": r.get("feret"),
        "minFeret": r.get("minFeret"),
        "feretAngle": r.get("feretAngle"),
        "feretX": r.get("feretX"),
        "feretY": r.get("feretY"),
        "x": r.get("x"),
        "y": r.get("y"),
        "xm": r.get("xm"),
        "ym": r.get("ym"),
        "centroidToCom": r.get("centroidToCom"),
        "bx": r.get("bx"),
        "by": r.get("by"),
        "width": r.get("width"),
        "height": r.get("height"),
        "mean": r.get("mean"),
        "median": r.get("median"),
        "mode": r.get("mode"),
        "min": r.get("min"),
        "max": r.get("max"),
        "stdDev": r.get("stdDev"),
        "skew": r.get("skew"),
        "kurt": r.get("kurt"),
        "rawIntDen": r.get("rawIntDen"),
        "intDen": r.get("intDen"),
        "corrTotalInt": r.get("corrTotalInt"),
        "corrMeanInt": r.get("corrMeanInt"),
        "corrMinInt": r.get("corrMinInt"),
        "corrMaxInt": r.get("corrMaxInt"),
        "bgRing": r.get("bgRing"),

        # timings
        "upload_s": r.get("upload_s"),
        "analyze_s": r.get("analyze_s"),
        "calculation_s": r.get("calculation_s"),
        "decode_rgb_s": r.get("decode_rgb_s"),
        "total_request_s": r.get("total_request_s"),

        # raw dumps
        "results_json": payload.get("results"),
        "params_json": None,   # optionally capture the params you used
        "profile_json": payload.get("profile"),
    }


# ----------------------------
# Endpoint presets (Option A)
# ----------------------------

def brightfield_defaults() -> Dict[str, Any]:
    return dict(
        sigma_pre=6.4,
        dilate_iter=4,
        erode_iter=5,
        min_area_px=60_000,
        max_area_px=20_000_000,
        min_circ=0.28,
        edge_margin=0.20,
        pixel_size_um=0.86,
        overlay_width=11,
        return_images=True,
        crop_overlay=False,
        crop_border_px=2,
        ring_px=20,
        invert_for_intensity=True,
        exclude_edge_particles=True,
        select_strategy="largest",
        area_filter_px=None,
        background_mode="ring",
        object_is_dark=True,
        ignore_zero_bins_for_mode_min=False,  # keep BF unchanged
    )

def fluorescence_defaults() -> Dict[str, Any]:
    return dict(
        sigma_pre=14,
        dilate_iter=10,
        erode_iter=8,
        min_area_px=1_000,
        max_area_px=10_000_000,
        min_circ=0.0,
        edge_margin=0.0,
        pixel_size_um=1.0,
        overlay_width=11,
        return_images=True,
        crop_overlay=False,
        crop_border_px=2,
        ring_px=20,
        invert_for_intensity=False,
        exclude_edge_particles=False,
        select_strategy="composite_filtered",
        area_filter_px=33_000,
        background_mode="inverse_of_composite",
        object_is_dark=False,
        ignore_zero_bins_for_mode_min=True,   # FL: ignore zero-bin for mode/min
    )

# ----------------------------
# API routes (Option A)
# ----------------------------

@api.post("/analyze/brightfield")
async def analyze_brightfield(
    file: UploadFile = File(...),
    sigma_pre: float = Query(brightfield_defaults()["sigma_pre"], ge=0.0),
    dilate_iter: int = Query(brightfield_defaults()["dilate_iter"], ge=0),
    erode_iter: int = Query(brightfield_defaults()["erode_iter"], ge=0),
    min_area_px: float = Query(brightfield_defaults()["min_area_px"], ge=0),
    min_circ: float   = Query(brightfield_defaults()["min_circ"], ge=0.0, le=1.0),
    edge_margin: float = Query(brightfield_defaults()["edge_margin"], ge=0.0, le=0.49),
    pixel_size_um: float = Query(brightfield_defaults()["pixel_size_um"], gt=0.0),
    return_images: bool = Query(brightfield_defaults()["return_images"]),
    profile: bool = Query(False),
    day0_area: Optional[float] = Query(None),
    run_id: Optional[str] = Query(None),
    day: Optional[str] = Query(None),
    organoid_number: Optional[str] = Query(None),
) -> Dict[str, Any]:
    timings: Dict[str, float] = {}

    with timed(timings, "upload_read_s"), time_block("read upload bytes"):
        data = await file.read()
    filename = file.filename or ""
    logger.info(f"Received BF file: {filename!r} ({len(data)} bytes)")

    # Ensure day and organoid_number are strings or None
    day = str(day) if day is not None else None
    organoid_number = str(organoid_number) if organoid_number is not None else None

    with timed(timings, "decode_rgb_s"), time_block("PIL decode + to RGB"):
        try:
            img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        except Exception as e:
            logger.exception(f"Invalid image payload (BF): {e}")
            raise HTTPException(400, f"Invalid image: {str(e)}")

    params = brightfield_defaults()
    params.update(
        sigma_pre=sigma_pre,
        dilate_iter=dilate_iter,
        erode_iter=erode_iter,
        min_area_px=min_area_px,
        min_circ=min_circ,
        edge_margin=edge_margin,
        pixel_size_um=pixel_size_um,
        return_images=return_images,
    )

    prof = None
    try:
        logger.info("Starting analysis (BF)")
        with ResourceProfiler("analyze_brightfield") as prof:
            with timed(timings, "analyze_total_s"), time_block("analyze_image total"):
                payload = analyze_image(img, **params)
        logger.info("Analysis completed successfully (BF)")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Analysis failed (BF): {e}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")

    try:
        logger.info("Starting postprocessing (BF)")
        with timed(timings, "postprocess_s"), time_block("growth-rate compute"):
            area_value = float(payload["results"]["area"])
            growth_rate = None
            if day == "0" or day == 0:
                growth_rate = 1.0
            else:
                if day0_area is not None and day0_area > 0:
                    growth_rate = area_value / float(day0_area)

            payload["results"]["day"] = day
            payload["results"]["organoidNumber"] = organoid_number
            payload["results"]["growthRate"] = growth_rate
            payload["results"]["type"] = "brightfield"

            if profile and prof and prof.metrics:
                payload["profile"] = prof.metrics

            payload["results"].update({
                "upload_s": timings.get("upload_read_s"),
                "analyze_s": timings.get("analyze_total_s"),
                "calculation_s": timings.get("postprocess_s"),
                "decode_rgb_s": timings.get("decode_rgb_s"),
                "total_request_s": round(sum(v for v in timings.values() if isinstance(v, (int, float))), 6)
            })
        logger.info("Postprocessing completed (BF)")
    except Exception as e:
        logger.exception(f"Postprocessing failed (BF): {e}")
        raise HTTPException(500, f"Postprocessing failed: {str(e)}")

    # Log before returning to help debug
    roi_size = len(payload.get("roi_image", "")) if payload.get("roi_image") else 0
    mask_size = len(payload.get("mask_image", "")) if payload.get("mask_image") else 0
    logger.info(f"Analysis complete, returning response (BF). Payload keys: {list(payload.keys())}, ROI size: {roi_size}, Mask size: {mask_size}")
    try:
        return payload
    except Exception as e:
        logger.exception(f"Unexpected error returning payload (BF): {e}")
        raise


@api.post("/analyze/fluorescence")
async def analyze_fluorescence(
    file: UploadFile = File(...),
    sigma_pre: float = Query(fluorescence_defaults()["sigma_pre"], ge=0.0),
    dilate_iter: int = Query(fluorescence_defaults()["dilate_iter"], ge=0),
    erode_iter: int = Query(fluorescence_defaults()["erode_iter"], ge=0),
    area_filter_px: float = Query(fluorescence_defaults()["area_filter_px"], ge=0),
    pixel_size_um: float = Query(fluorescence_defaults()["pixel_size_um"], gt=0.0),
    return_images: bool = Query(fluorescence_defaults()["return_images"]),
    profile: bool = Query(False),
    day0_area: Optional[float] = Query(None),
    run_id: Optional[str] = Query(None),
    day: Optional[str] = Query(None),
    organoid_number: Optional[str] = Query(None),
) -> Dict[str, Any]:
    timings: Dict[str, float] = {}

    with timed(timings, "upload_read_s"), time_block("read upload bytes"):
        data = await file.read()
    filename = file.filename or ""
    logger.info(f"Received FL file: {filename!r} ({len(data)} bytes)")

    # Ensure day and organoid_number are strings or None
    day = str(day) if day is not None else None
    organoid_number = str(organoid_number) if organoid_number is not None else None

    with timed(timings, "decode_rgb_s"), time_block("PIL decode + to RGB"):
        try:
            img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        except Exception as e:
            logger.exception(f"Invalid image payload (FL): {e}")
            raise HTTPException(400, f"Invalid image: {str(e)}")

    params = fluorescence_defaults()
    params.update(
        sigma_pre=sigma_pre,
        dilate_iter=dilate_iter,
        erode_iter=erode_iter,
        area_filter_px=area_filter_px,
        return_images=return_images,
    )

    prof = None
    try:
        logger.info("Starting analysis (FL)")
        with ResourceProfiler("analyze_fluorescence") as prof:
            with timed(timings, "analyze_total_s"), time_block("analyze_image total"):
                payload = analyze_image(img, **params)
        logger.info("Analysis completed successfully (FL)")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Analysis failed (FL): {e}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")

    try:
        logger.info("Starting postprocessing (FL)")
        with timed(timings, "postprocess_s"), time_block("growth-rate compute"):
            area_value = float(payload["results"]["area"])
            growth_rate = None
            
            logger.debug(f"FL growth-rate compute | day={day} day0_area={day0_area} area_value={area_value}")

            if day == "0" or day == 0:
                growth_rate = 1.0
            else:
                if day0_area is not None and day0_area > 0:
                    growth_rate = area_value / float(day0_area)

            logger.debug(f"Computed growth_rate={growth_rate}")

            payload["results"]["day"] = day
            payload["results"]["organoidNumber"] = organoid_number
            payload["results"]["growthRate"] = growth_rate
            payload["results"]["type"] = "fluorescence"   # bugfix

            if profile and prof and prof.metrics:
                payload["profile"] = prof.metrics

            payload["results"].update({
                "upload_s": timings.get("upload_read_s"),
                "analyze_s": timings.get("analyze_total_s"),
                "calculation_s": timings.get("postprocess_s"),
                "decode_rgb_s": timings.get("decode_rgb_s"),
                "total_request_s": round(sum(v for v in timings.values() if isinstance(v, (int, float))), 6)
            })
        logger.info("Postprocessing completed (FL)")
    except Exception as e:
        logger.exception(f"Postprocessing failed (FL): {e}")
        raise HTTPException(500, f"Postprocessing failed: {str(e)}")

    # Log before returning to help debug
    roi_size = len(payload.get("roi_image", "")) if payload.get("roi_image") else 0
    mask_size = len(payload.get("mask_image", "")) if payload.get("mask_image") else 0
    logger.info(f"Analysis complete, returning response (FL). Payload keys: {list(payload.keys())}, ROI size: {roi_size}, Mask size: {mask_size}")
    try:
        return payload
    except Exception as e:
        logger.exception(f"Unexpected error returning payload (FL): {e}")
        raise


async def persist_one_result(run_id: str, filename: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    bucket = "orgprofiler"
    base = f"runs/{run_id}/{filename or 'image'}.png"
    roi_path  = base.replace(".png", ".roi.png")
    mask_path = base.replace(".png", ".mask.png")

    async with StorageUploader(settings.SUPABASE_URL, settings.SUPABASE_KEY) as up:
        roi_url  = await up.upload_png_dataurl(bucket, roi_path,  payload["roi_image"])
        mask_url = await up.upload_png_dataurl(bucket, mask_path, payload["mask_image"])

    analysis_type = payload.get("results", {}).get("type", "unknown")
    row = _result_row_from_payload(run_id, filename, payload, analysis_type=analysis_type)
    row["roi_image_path"] = roi_url
    row["mask_image_path"] = mask_url
    return row

class PersistItem(BaseModel):
    filename: str
    payload: Dict[str, Any] 

import asyncio

@api.post("/runs/{run_id}/persist")
async def persist_run_results(run_id: str, items: List[PersistItem]):
    if not items:
        return {"inserted": 0}

    # persist_one_result(...) is async (it uses anyio.to_thread inside)
    rows = await asyncio.gather(*[
        persist_one_result(run_id, it.filename, it.payload) for it in items
    ])

    ins = supabase.table("analysis_results").insert(rows).execute()
    if getattr(ins, "error", None):
        raise HTTPException(500, f"DB insert failed: {ins.error}")

    supabase.table("analysis_runs").update({"status": "running"}).eq("id", run_id).execute()
    return {"inserted": len(rows)}


# ----------------------------
# Request timing middleware
# ----------------------------
@api.middleware("http")
async def log_request_timing(request, call_next):
    t0 = time.perf_counter()
    response = None
    status_code = "?"
    try:
        response = await call_next(request)
        status_code = getattr(response, "status_code", "?")
        return response
    except Exception:
        # Make sure we still log timing; keep original stack trace
        status_code = "EXC"
        raise
    finally:
        dt = time.perf_counter() - t0
        clen_req = request.headers.get("content-length")
        # response can be None on exception; avoid touching response.headers
        logger.info(
            f"[HTTP] {request.method} {request.url.path} -> {status_code} "
            f"in {dt:.3f}s (Content-Length={clen_req})"
        )



# ----------------------------
# RUNS
# ----------------------------


@api.post("/runs")
def create_run(
    name: str = Body(embed=True),
    user_id: Optional[str] = Body(default=None, embed=True),  # if you track per-user
):
    run_id = str(uuid4())
    # Optional: client can supply run_id for idempotency; else we generate above.
    data = {
        "id": run_id,
        "name": name,
        "user_id": user_id,
        "status": "pending",
    }
    res = supabase.table("analysis_runs").insert(data).execute()
    if getattr(res, "error", None):
        raise HTTPException(500, f"Failed to create run: {res.error.message}")
    return {"run_id": run_id, "status": "pending"}


class RunStatusBody(BaseModel):
    status: Literal["running","completed","failed","canceled"]

@api.post("/runs/{run_id}/status")
def set_run_status(run_id: str, body: RunStatusBody):
    status = body.status
    patch = {"status": status}
    if status in ("completed", "failed", "canceled"):
        patch["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res = supabase.table("analysis_runs").update(patch).eq("id", run_id).execute()
    if hasattr(res, "error") and res.error:
        raise HTTPException(500, f"DB error: {res.error}")
    return {"ok": True}


# ----------------------------
# Supabase Storage uploader (async)
# ----------------------------
class StorageUploader:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.base = supabase_url.rstrip("/")
        self.key  = supabase_key
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            http2=False,
            timeout=httpx.Timeout(connect=20, read=60, write=60, pool=20),
            limits=httpx.Limits(max_keepalive_connections=0, max_connections=10),
            headers={"Connection": "close", "Authorization": f"Bearer {self.key}"}
        )
        return self

    async def __aexit__(self, *exc):
        if self._client:
            await self._client.aclose()

    async def upload_png_dataurl(self, bucket: str, path: str, data_url: str) -> str:
        assert self._client is not None
        comma = data_url.find(",")
        if comma == -1:
            raise ValueError("Invalid data URL")
        body = base64.b64decode(data_url[comma+1:], validate=True)

        # Supabase Storage REST: POST /storage/v1/object/{bucket}/{path}
        # Docs: https://supabase.com/docs/reference/storage/createobject
        url = f"{self.base}/storage/v1/object/{bucket}/{path}"
        resp = await self._client.post(
            url,
            headers={"Content-Type": "image/png", "x-upsert": "true"},
            content=body,
        )
        resp.raise_for_status()

        # get public URL
        pub = f"{self.base}/storage/v1/object/public/{bucket}/{path}"
        return pub