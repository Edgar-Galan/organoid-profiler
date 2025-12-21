import math
import numpy as np
from typing import Dict, Any, Optional, Literal
from scipy import ndimage as ndi
from skimage import measure, segmentation, morphology
from skimage.morphology import disk
from skimage.measure import perimeter_crofton
from loguru import logger

from .imaging import convert_rgb_to_grayscale_uint8, convert_array_to_data_url_png
from .profiling import time_block, ResourceProfiler
from .segmentation import build_segmentation_mask_fiji_style, build_segmentation_mask_cpsam
from .metrics import calculate_feret_features_from_points
from .background import estimate_background_from_ring

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
    invert_for_intensity: bool,
    exclude_edge_particles: bool,
    select_strategy: Literal["largest", "composite_filtered"],
    area_filter_px: Optional[float],
    background_mode: Literal["ring", "inverse_of_composite"],
    object_is_dark: bool,
    ignore_zero_bins_for_mode_min: bool = False,
    segmentation_method: Literal["fiji", "cpsam"] = "fiji",
    # Cellpose 4 / Custom model parameters
    cellpose_model_type: str = "cpsam",
    cellpose_pretrained_model: Optional[str] = None,
    cellpose_diameter: Optional[float] = None,
    cellpose_flow_threshold: float = 0.4,
    cellpose_cellprob_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Core analysis logic for both brightfield and fluorescence images."""
    H, W, _ = img.shape
    
    if crop_border_px > 0 and min(img.shape[:2]) > 2 * crop_border_px:
        img = img[crop_border_px:-crop_border_px, crop_border_px:-crop_border_px, :]
        H, W, _ = img.shape

    flow_image_base64 = ""
    with time_block("segmentation"):
        if segmentation_method == "cpsam":
            res = build_segmentation_mask_cpsam(
                image_rgb=img,
                return_flows=return_images,
                model_type=cellpose_model_type,
                pretrained_model=cellpose_pretrained_model,
                diameter=cellpose_diameter,
                flow_threshold=cellpose_flow_threshold,
                cellprob_threshold=cellpose_cellprob_threshold,
            )
            if isinstance(res, tuple):
                mask_bool, flow_img = res
                if return_images:
                    flow_image_base64 = convert_array_to_data_url_png(flow_img)
            else:
                mask_bool = res
        else:
            mask_bool = build_segmentation_mask_fiji_style(
                image_rgb=img,
                gaussian_sigma=sigma_pre,
                dilation_iterations=dilate_iter,
                erosion_iterations=erode_iter,
                object_is_dark=object_is_dark,
            )

    labels = measure.label(mask_bool, connectivity=2)
    if labels.max() == 0:
        raise ValueError("No organoids found in the image")

    xmin, xmax = W * edge_margin, W * (1.0 - edge_margin)
    ymin, ymax = H * edge_margin, H * (1.0 - edge_margin)

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

    if area_filter_px is not None and keep_mask.any():
        lab2 = measure.label(keep_mask, connectivity=2)
        keep2 = np.zeros_like(keep_mask, dtype=bool)
        for r in measure.regionprops(lab2):
            if float(r.area) >= float(area_filter_px):
                keep2 |= (lab2 == r.label)
        keep_mask = keep2

    if not keep_mask.any():
        largest = max(measure.regionprops(labels), key=lambda rr: rr.area)
        keep_mask = (labels == largest.label)

    if select_strategy == "largest":
        union_lab = measure.label(keep_mask, connectivity=2)
        regs = measure.regionprops(union_lab)
        props = max(regs, key=lambda r: r.area)
        mask_measured = (union_lab == props.label)
        major_px = float(props.major_axis_length or 0.0)
        minor_px = float(props.minor_axis_length or 0.0)
        angle = float(np.degrees(props.orientation or 0.0))
        solidity = float(getattr(props, "solidity", 0.0))
        minr, minc, maxr, maxc = props.bbox
        cy_px, cx_px = props.centroid
    else:
        mask_measured = keep_mask
        union_lbl = mask_measured.astype(np.uint8)
        rp = measure.regionprops(union_lbl)[0]
        minr, minc, maxr, maxc = rp.bbox
        cy_px, cx_px = rp.centroid
        major_px = float(rp.major_axis_length or 0.0)
        minor_px = float(rp.minor_axis_length or 0.0)
        angle = float(np.degrees(rp.orientation or 0.0))
        solidity = float(getattr(rp, "solidity", 0.0))

    area_px = float(mask_measured.sum())
    perim_px = float(perimeter_crofton(mask_measured, directions=4))

    contours = measure.find_contours(mask_measured, 0.5)
    if contours:
        cont = max(contours, key=lambda c: c.shape[0])
        if cont.shape[0] > 4000:
            cont = cont[::4]
        pts = np.column_stack((cont[:, 1], cont[:, 0]))
    else:
        pts = np.empty((0, 2))

    if pts.shape[0] >= 3:
        feret_px, minFeret_px, feretAngle, feretX_px, feretY_px = calculate_feret_features_from_points(pts)
    else:
        feret_px = minFeret_px = feretAngle = feretX_px = feretY_px = 0.0

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

    grayscale_uint8 = convert_rgb_to_grayscale_uint8(img)
    intensity_image = (255 - grayscale_uint8).astype(np.uint8) if invert_for_intensity else grayscale_uint8
    intensity_values = intensity_image[mask_measured].astype(np.uint8)
    
    mean_intensity = float(intensity_values.mean())
    median_intensity = float(np.median(intensity_values))
    histogram = np.bincount(intensity_values, minlength=256)
    
    if ignore_zero_bins_for_mode_min and histogram[1:].sum() > 0:
        mode_intensity = float(histogram[1:].argmax() + 1)
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
    center_of_mass_y_px, center_of_mass_x_px = ndi.center_of_mass(
        intensity_image, labels=mask_measured.astype(np.uint8), index=1
    )
    center_of_mass_x, center_of_mass_y = center_of_mass_x_px * px_um, center_of_mass_y_px * px_um

    if background_mode == "ring":
        background_intensity = estimate_background_from_ring(
            intensity_image, mask_measured, ring_width_pixels=ring_px, method="median",
            bounding_box=(int(by_px), int(bx_px), int(by_px + height_px), int(bx_px + width_px))
        )
    else:
        roi_intensity = intensity_image[int(by_px):int(by_px+height_px), int(bx_px):int(bx_px+width_px)]
        roi_background_mask = ~mask_measured[int(by_px):int(by_px+height_px), int(bx_px):int(bx_px+width_px)]
        background_values = roi_intensity[roi_background_mask]
        background_intensity = float(np.median(background_values)) if background_values.size else 0.0

    corrected_total_intensity = raw_integrated_density - background_intensity * area_px
    corrected_mean_intensity = mean_intensity - background_intensity
    corrected_min_intensity = (min_intensity_positive if ignore_zero_bins_for_mode_min else min_intensity) - background_intensity
    corrected_max_intensity = max_intensity - background_intensity
    equivalent_diameter = math.sqrt(4.0 * area / math.pi)
    centroid_to_center_of_mass_distance = math.hypot(center_of_mass_x - cx, center_of_mass_y - cy)

    roi_image_base64 = ""
    mask_image_base64 = ""
    if return_images:
        if crop_overlay:
            cropped_mask = mask_measured[int(by_px):int(by_px+height_px), int(bx_px):int(bx_px+width_px)]
            overlay_image = img[int(by_px):int(by_px+height_px), int(bx_px):int(bx_px+width_px)].copy()
        else:
            cropped_mask = mask_measured
            overlay_image = img.copy()
        
        boundary_pixels = segmentation.find_boundaries(cropped_mask, mode="outer")
        dilated_boundaries = morphology.binary_dilation(boundary_pixels, disk(max(1, overlay_width // 2)))
        overlay_image[dilated_boundaries] = np.array([255, 0, 255], dtype=np.uint8)
        
        roi_image_base64 = convert_array_to_data_url_png(overlay_image)
        mask_image_base64 = convert_array_to_data_url_png((cropped_mask.astype(np.uint8) * 255))

    return {
        "results": {
            "area": area, "mean": mean_intensity, "stdDev": std_dev_intensity, "mode": mode_intensity,
            "min": min_intensity, "max": max_intensity, "x": cx, "y": cy, "xm": center_of_mass_x,
            "ym": center_of_mass_y, "perim": perim, "bx": bx, "by": by, "width": width, "height": height,
            "major": major, "minor": minor, "angle": angle, "circ": circ, "feret": feret,
            "intDen": raw_integrated_density, "median": median_intensity, "skew": -skewness,
            "kurt": kurtosis, "rawIntDen": raw_integrated_density, "feretX": feretX, "feretY": feretY,
            "feretAngle": feretAngle, "minFeret": minFeret, "ar": ar, "round": roundness,
            "solidity": solidity, "eqDiam": equivalent_diameter, "corrTotalInt": corrected_total_intensity,
            "corrMeanInt": corrected_mean_intensity, "corrMinInt": corrected_min_intensity,
            "corrMaxInt": corrected_max_intensity, "centroidToCom": centroid_to_center_of_mass_distance,
            "bgRing": background_intensity,
        },
        "roi_image": roi_image_base64,
        "mask_image": mask_image_base64,
        "flow_image": flow_image_base64,
    }

