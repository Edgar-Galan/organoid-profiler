# server.py
from __future__ import annotations
import datetime
import json
import io
import os
import time
import asyncio, zipfile, httpx, anyio
from typing import Tuple, Optional, Literal, Union, List, Dict, Any
from uuid import uuid4

import numpy as np
from PIL import Image
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger
import dotenv

# Import from our new package
from organoid_profiler import (
    analyze_image,
    ResourceProfiler,
    timed,
    time_block
)
from organoid_profiler.io.storage import StorageUploader
from organoid_profiler.io.database import get_supabase_client, format_analysis_result

dotenv.load_dotenv()

# Global queue for analysis jobs to prevent memory exhaustion
analysis_queue = asyncio.Queue()

# ----------------------------
# Settings & Clients
# ----------------------------

class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SUPABASE_BUCKET: str = "orgprofiler"

    @field_validator("SUPABASE_URL", "SUPABASE_KEY", mode="before")
    @classmethod
    def strip_spaces(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip().strip("'").strip('"')
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False, 
    )

settings = Settings()
# Debug: check if keys are loaded correctly (first 5 chars only for security)
logger.info(f"Settings loaded. URL: {settings.SUPABASE_URL[:10]}... Key starts with: {settings.SUPABASE_KEY[:5]}...")
supabase = get_supabase_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

# ----------------------------
# FastAPI setup
# ----------------------------
api = FastAPI(title="Organoid Profiler API")

# Add global exception handler for clearer error messages
@api.exception_handler(Exception)
async def global_exception_handler(request, exc):
    if isinstance(exc, HTTPException):
        return await http_exception_handler(request, exc)
    logger.exception(f"Unhandled exception in {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

from fastapi.exception_handlers import http_exception_handler

api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://organoid-profiler.com",
        "https://www.organoid-profiler.com",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api.get("/healthz")
def healthz():
    return {"ok": True}

@api.get("/healthz/db")
def healthz_db():
    try:
        supabase.table("analysis_runs").select("id").limit(1).execute()
        return {"ok": True, "database": "connected"}
    except Exception as e:
        logger.exception("Database health check failed")
        raise HTTPException(503, f"Database connection failed: {str(e)}")

# ----------------------------
# Presets
# ----------------------------

def is_baseline_day(day: Any) -> bool:
    """Check if a day string or value represents Day 0/Baseline."""
    return str(day).lower() in ("0", "00", "0.0", "0.00", "d00", "d0", "d0.0")

def brightfield_defaults() -> Dict[str, Any]:
    return dict(
        sigma_pre=6, dilate_iter=3, erode_iter=3, min_area_px=50_000,
        max_area_px=750_000, min_circ=0.38, edge_margin=0.10,
        pixel_size_um=1, overlay_width=10, return_images=True,
        crop_overlay=False, crop_border_px=2, ring_px=20,
        invert_for_intensity=True, exclude_edge_particles=True,
        select_strategy="all", area_filter_px=None,
        background_mode="ring", object_is_dark=True,
        ignore_zero_bins_for_mode_min=False,
        # Watershed defaults for FIJI-style segmentation (conservative to avoid oversegmentation)
        watershed_enabled=True,
        watershed_min_distance_px=15,
        watershed_apply_below_min_circ=True,
        watershed_apply_above_max_area=True,
        cellpose_model_type="cpsam",
        cellpose_pretrained_model="organoid_cpsam",
        cellpose_diameter=None,
        cellpose_flow_threshold=0.4,
        cellpose_cellprob_threshold=0.0,
    )

def fluorescence_defaults() -> Dict[str, Any]:
    return dict(
        sigma_pre=1, dilate_iter=1, erode_iter=1, min_area_px=1_000,
        max_area_px=10_000_000, min_circ=0.0, edge_margin=0.0,
        pixel_size_um=1.0, overlay_width=10, return_images=True,
        crop_overlay=False, crop_border_px=2, ring_px=20,
        invert_for_intensity=False, exclude_edge_particles=False,
        select_strategy="composite_filtered", area_filter_px=33_000,
        background_mode="inverse_of_composite", object_is_dark=False,
        ignore_zero_bins_for_mode_min=True,
        # Watershed for FIJI-style fluorescence (disabled by default to be safe)
        watershed_enabled=False,
        watershed_min_distance_px=15,
        watershed_apply_below_min_circ=True,
        watershed_apply_above_max_area=True,
        cellpose_model_type="cpsam",
        cellpose_pretrained_model="organoid_cpsam",
        cellpose_diameter=None,
        cellpose_flow_threshold=0.4,
        cellpose_cellprob_threshold=0.0,
    )

# ----------------------------
# API routes
# ----------------------------

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def run_analysis(img: np.ndarray, params: Dict[str, Any], timings: Dict[str, float], 
                       label: str, profile: bool, day: Optional[str], day0_area: Optional[float],
                       organoid_number: Optional[str], analysis_type: str, 
                       segmentation_method: str = "fiji") -> Dict[str, Any]:
    prof = None
    logger.info(f"STARTING ANALYSIS: Type={analysis_type}, Method={segmentation_method}, Day={day}, Organoid={organoid_number}")
    try:
        with ResourceProfiler(f"analyze_{analysis_type}") as prof:
            with timed(timings, "analyze_total_s"):
                # Ensure segmentation_method is not duplicated in params
                params.pop("segmentation_method", None)
                # Run the heavy CPU-bound analysis in a thread to avoid blocking the event loop
                from functools import partial
                func = partial(analyze_image, img, segmentation_method=segmentation_method, **params)
                payload = await anyio.to_thread.run_sync(func)
        
        with timed(timings, "postprocess_s"):
            # Handle both list (all ROIs) and dict (single ROI) formats for backward compatibility
            results = payload["results"]
            is_list_format = isinstance(results, list)
            
            if is_list_format:
                # Process each ROI in the list
                processed_results = []
                for idx, roi_result in enumerate(results):
                    area_value = float(roi_result["area"])
                    
                    # Robust Day 0 check
                    is_baseline = is_baseline_day(day)
                    
                    if is_baseline:
                        growth_rate = 1.0
                    elif day0_area:
                        try:
                            growth_rate = float(area_value) / float(day0_area)
                        except (ValueError, ZeroDivisionError, TypeError):
                            growth_rate = None
                    else:
                        growth_rate = None
                    
                    roi_result.update({
                        "day": str(day) if day is not None else "0",
                        "organoidNumber": str(organoid_number) if organoid_number is not None else "1",
                        "roiIndex": idx + 1,
                        "growth_rate": growth_rate,
                        "type": analysis_type,
                        "analyze_s": timings.get("analyze_total_s"),
                        "calculation_s": timings.get("postprocess_s"),
                        "decode_rgb_s": timings.get("decode_rgb_s"),
                        "total_request_s": round(sum(v for v in timings.values() if isinstance(v, (int, float))), 6)
                    })
                    processed_results.append(roi_result)
                
                payload["results"] = processed_results
                # Log summary with largest ROI for compatibility
                largest_roi = max(processed_results, key=lambda r: r.get("area", 0))
                total_area = sum(r.get("area", 0) for r in processed_results)
                logger.info(f"ANALYSIS SUCCESS: {label} | {len(processed_results)} ROIs | Total Area={total_area:.2f} | Largest Area={largest_roi.get('area', 0):.2f} | Growth={largest_roi.get('growth_rate')} | TotalTime={timings.get('analyze_total_s'):.3f}s")
            else:
                # Backward compatibility: single ROI format
                area_value = float(results["area"])
                
                # Robust Day 0 check
                is_baseline = is_baseline_day(day)
                
                if is_baseline:
                    growth_rate = 1.0
                elif day0_area:
                    try:
                        growth_rate = float(area_value) / float(day0_area)
                    except (ValueError, ZeroDivisionError, TypeError):
                        growth_rate = None
                else:
                    growth_rate = None

                payload["results"].update({
                    "day": str(day) if day is not None else "0",
                    "organoidNumber": str(organoid_number) if organoid_number is not None else "1",
                    "growth_rate": growth_rate,
                    "type": analysis_type,
                    "analyze_s": timings.get("analyze_total_s"),
                    "calculation_s": timings.get("postprocess_s"),
                    "decode_rgb_s": timings.get("decode_rgb_s"),
                    "total_request_s": round(sum(v for v in timings.values() if isinstance(v, (int, float))), 6)
                })
                logger.info(f"ANALYSIS SUCCESS: {label} | Area={payload['results'].get('area'):.2f} | Growth={growth_rate} | TotalTime={timings.get('analyze_total_s'):.3f}s")
            
            if profile and prof:
                payload["profile"] = prof.metrics
        return payload
    except ValueError as e:
        logger.warning(f"Analysis validation error ({analysis_type}): {e}")
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.exception(f"Analysis failed ({analysis_type}): {e}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")

@api.post("/analyze/brightfield")
async def analyze_brightfield(
    file: UploadFile = File(...),
    sigma_pre: float = Query(brightfield_defaults()["sigma_pre"]),
    dilate_iter: int = Query(brightfield_defaults()["dilate_iter"]),
    erode_iter: int = Query(brightfield_defaults()["erode_iter"]),
    min_area_px: float = Query(brightfield_defaults()["min_area_px"]),
    min_circ: float = Query(brightfield_defaults()["min_circ"]),
    edge_margin: float = Query(brightfield_defaults()["edge_margin"]),
    pixel_size_um: float = Query(brightfield_defaults()["pixel_size_um"]),
    return_images: bool = Query(True),
    profile: bool = Query(False),
    day0_area: Optional[float] = Query(None),
    day: Optional[str] = Query(None),
    organoid_number: Optional[str] = Query(None),
    segmentation_method: str = Query("fiji"),
    cellpose_model_type: str = Query("cpsam"),
    cellpose_pretrained_model: Optional[str] = Query("organoid_cpsam"),
    cellpose_diameter: Optional[float] = Query(None),
    cellpose_flow_threshold: float = Query(0.4),
    cellpose_cellprob_threshold: float = Query(0.0),
):
    timings = {}
    with timed(timings, "upload_read_s"):
        data = await file.read()
    
    with timed(timings, "decode_rgb_s"):
        img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))

    params = brightfield_defaults()
    params.update(
        sigma_pre=sigma_pre, dilate_iter=dilate_iter, erode_iter=erode_iter,
        min_area_px=min_area_px, min_circ=min_circ, edge_margin=edge_margin,
        pixel_size_um=pixel_size_um, return_images=return_images,
        cellpose_model_type=cellpose_model_type,
        cellpose_pretrained_model=cellpose_pretrained_model,
        cellpose_diameter=cellpose_diameter,
        cellpose_flow_threshold=cellpose_flow_threshold,
        cellpose_cellprob_threshold=cellpose_cellprob_threshold
    )

    return await run_analysis(img, params, timings, "BF", profile, day, day0_area, organoid_number, "brightfield", segmentation_method=segmentation_method)

@api.post("/analyze/fluorescence")
async def analyze_fluorescence(
    file: UploadFile = File(...),
    sigma_pre: float = Query(fluorescence_defaults()["sigma_pre"]),
    dilate_iter: int = Query(fluorescence_defaults()["dilate_iter"]),
    erode_iter: int = Query(fluorescence_defaults()["erode_iter"]),
    area_filter_px: float = Query(fluorescence_defaults()["area_filter_px"]),
    pixel_size_um: float = Query(fluorescence_defaults()["pixel_size_um"]),
    return_images: bool = Query(True),
    profile: bool = Query(False),
    day0_area: Optional[float] = Query(None),
    day: Optional[str] = Query(None),
    organoid_number: Optional[str] = Query(None),
    segmentation_method: str = Query("fiji"),
        cellpose_model_type: str = Query("cpsam"),
    cellpose_pretrained_model: Optional[str] = Query("organoid_cpsam"),
    cellpose_diameter: Optional[float] = Query(None),
    cellpose_flow_threshold: float = Query(0.4),
    cellpose_cellprob_threshold: float = Query(0.0),
):
    timings = {}
    with timed(timings, "upload_read_s"):
        data = await file.read()
    
    with timed(timings, "decode_rgb_s"):
            img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))

    params = fluorescence_defaults()
    params.update(
        sigma_pre=sigma_pre, dilate_iter=dilate_iter, erode_iter=erode_iter,
        area_filter_px=area_filter_px, pixel_size_um=pixel_size_um, return_images=return_images,
        cellpose_model_type=cellpose_model_type,
        cellpose_pretrained_model=cellpose_pretrained_model,
        cellpose_diameter=cellpose_diameter,
        cellpose_flow_threshold=cellpose_flow_threshold,
        cellpose_cellprob_threshold=cellpose_cellprob_threshold
    )

    return await run_analysis(img, params, timings, "FL", profile, day, day0_area, organoid_number, "fluorescence", segmentation_method=segmentation_method)

@api.post("/analyze")
async def analyze_unified(
    file: UploadFile = File(...),
    type: str = Query("brightfield"),
    sigma_pre: Optional[float] = Query(None),
    dilate_iter: Optional[int] = Query(None),
    erode_iter: Optional[int] = Query(None),
    min_area_px: Optional[float] = Query(None),
    min_circ: Optional[float] = Query(None),
    edge_margin: Optional[float] = Query(None),
    pixel_size_um: Optional[float] = Query(None),
    area_filter_px: Optional[float] = Query(None),
    return_images: bool = Query(True),
    profile: bool = Query(False),
    day0_area: Optional[float] = Query(None),
    day: Optional[str] = Query(None),
    organoid_number: Optional[str] = Query(None),
    segmentation_method: str = Query("fiji"),
    cellpose_model_type: str = Query("cpsam"),
    cellpose_pretrained_model: Optional[str] = Query("organoid_cpsam"),
    cellpose_diameter: Optional[float] = Query(None),
    cellpose_flow_threshold: float = Query(0.4),
    cellpose_cellprob_threshold: float = Query(0.0),
):
    if type == "brightfield":
        defaults = brightfield_defaults()
        return await analyze_brightfield(
            file, 
            sigma_pre=sigma_pre if sigma_pre is not None else defaults["sigma_pre"],
            dilate_iter=dilate_iter if dilate_iter is not None else defaults["dilate_iter"],
            erode_iter=erode_iter if erode_iter is not None else defaults["erode_iter"],
            min_area_px=min_area_px if min_area_px is not None else defaults["min_area_px"],
            min_circ=min_circ if min_circ is not None else defaults["min_circ"],
            edge_margin=edge_margin if edge_margin is not None else defaults["edge_margin"],
            pixel_size_um=pixel_size_um if pixel_size_um is not None else defaults["pixel_size_um"],
            return_images=return_images, profile=profile, day0_area=day0_area, day=day, organoid_number=organoid_number,
            segmentation_method=segmentation_method,
            cellpose_model_type=cellpose_model_type,
            cellpose_pretrained_model=cellpose_pretrained_model,
            cellpose_diameter=cellpose_diameter,
            cellpose_flow_threshold=cellpose_flow_threshold,
            cellpose_cellprob_threshold=cellpose_cellprob_threshold
        )
    else:
        defaults = fluorescence_defaults()
        return await analyze_fluorescence(
            file,
            sigma_pre=sigma_pre if sigma_pre is not None else defaults["sigma_pre"],
            dilate_iter=dilate_iter if dilate_iter is not None else defaults["dilate_iter"],
            erode_iter=erode_iter if erode_iter is not None else defaults["erode_iter"],
            area_filter_px=area_filter_px if area_filter_px is not None else defaults["area_filter_px"],
            pixel_size_um=pixel_size_um if pixel_size_um is not None else defaults["pixel_size_um"],
            return_images=return_images, profile=profile, day0_area=day0_area, day=day, organoid_number=organoid_number,
            segmentation_method=segmentation_method,
            cellpose_model_type=cellpose_model_type,
            cellpose_pretrained_model=cellpose_pretrained_model,
            cellpose_diameter=cellpose_diameter,
            cellpose_flow_threshold=cellpose_flow_threshold,
            cellpose_cellprob_threshold=cellpose_cellprob_threshold
        )

# ----------------------------
# Persistence
# ----------------------------

async def persist_one_result(run_id: str, filename: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    bucket = settings.SUPABASE_BUCKET
    # Remove existing extension and use underscores for cleaner paths
    base_name = os.path.splitext(filename)[0] if filename else "image"
    roi_path = f"runs/{run_id}/{base_name}_roi.png"
    mask_path = f"runs/{run_id}/{base_name}_mask.png"
    flow_path = f"runs/{run_id}/{base_name}_flow.png"

    async with StorageUploader(settings.SUPABASE_URL, settings.SUPABASE_KEY) as up:
        roi_url = None
        roi_img = payload.get("roi_image")
        if isinstance(roi_img, str) and "," in roi_img:
            try:
                roi_url = await up.upload_png_dataurl(bucket, roi_path, roi_img)
            except Exception as e:
                logger.error(f"Failed to upload ROI image: {e}")
        
        mask_url = None
        mask_img = payload.get("mask_image")
        if isinstance(mask_img, str) and "," in mask_img:
            try:
                mask_url = await up.upload_png_dataurl(bucket, mask_path, mask_img)
            except Exception as e:
                logger.error(f"Failed to upload Mask image: {e}")

        flow_url = None
        flow_img = payload.get("flow_image")
        if isinstance(flow_img, str) and "," in flow_img:
            try:
                flow_url = await up.upload_png_dataurl(bucket, flow_path, flow_img)
            except Exception as e:
                logger.error(f"Failed to upload Flow image: {e}")
    
    # Get analysis_type from results (handle both list and dict formats)
    results = payload.get("results", {})
    if isinstance(results, list):
        analysis_type = results[0].get("type", "unknown") if results else "unknown"
    else:
        analysis_type = results.get("type", "unknown")
    row = format_analysis_result(run_id, filename, payload, analysis_type)
    row.update({"roi_image_path": roi_url, "mask_image_path": mask_url})
    return row

class PersistItem(BaseModel):
    filename: str
    payload: Dict[str, Any] 

@api.post("/runs/{run_id}/persist")
async def persist_run_results(run_id: str, items: List[PersistItem]):
    if not items: return {"inserted": 0}
    rows = await asyncio.gather(*[persist_one_result(run_id, it.filename, it.payload) for it in items])
    res = supabase.table("analysis_results").insert(rows).execute()
    supabase.table("analysis_runs").update({"status": "running"}).eq("id", run_id).execute()
    return {"inserted": len(rows)}

# ----------------------------
# Runs Management
# ----------------------------

@api.post("/runs")
def create_run(name: str = Body(embed=True), user_id: Optional[str] = Body(default=None, embed=True), total_files: Optional[int] = Body(default=None, embed=True)):
    run_id = str(uuid4())
    data = {"id": run_id, "name": name, "user_id": user_id, "status": "pending"}
    if total_files is not None:
        # We'll try to include total_files but be ready for it to fail if column doesn't exist
        try:
            temp_data = data.copy()
            temp_data["total_files"] = total_files
            supabase.table("analysis_runs").insert(temp_data).execute()
            return {"run_id": run_id, "status": "pending"}
        except Exception as e:
            if "total_files" in str(e):
                logger.warning("total_files column missing in analysis_runs table, falling back to base insert")
            else:
                raise e
    
    supabase.table("analysis_runs").insert(data).execute()
    return {"run_id": run_id, "status": "pending"}

class RunStatusBody(BaseModel):
    status: Literal["running","completed","failed","canceled"]

@api.post("/runs/{run_id}/status")
def set_run_status(run_id: str, body: RunStatusBody):
    patch = {"status": body.status}
    if body.status in ("completed", "failed", "canceled"):
        patch["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    supabase.table("analysis_runs").update(patch).eq("id", run_id).execute()
    return {"ok": True}

@api.get("/runs")
def get_runs():
    # Use the view if available, or join manually
    try:
        res = supabase.table("analysis_runs_with_counts").select("*").order("created_at", desc=True).execute()
        return res.data
    except Exception:
        # Fallback to base table if view doesn't exist
        res = supabase.table("analysis_runs").select("*").order("created_at", desc=True).execute()
        return res.data

@api.get("/runs/{run_id}")
def get_run(run_id: str):
    res = supabase.table("analysis_runs").select("*, analysis_results(*)").eq("id", run_id).single().execute()
    return res.data

@api.get("/runs/{run_id}/status")
def get_run_status(run_id: str):
    run_res = supabase.table("analysis_runs").select("status, created_at, completed_at").eq("id", run_id).single().execute()
    if not run_res.data:
        raise HTTPException(404, "Run not found")
    
    count_res = supabase.table("analysis_results").select("id", count="exact").eq("run_id", run_id).execute()
    return {
        "run_id": run_id,
        "status": run_res.data["status"],
        "created_at": run_res.data["created_at"],
        "completed_at": run_res.data["completed_at"],
        "result_count": count_res.count or 0
    }

@api.get("/runs/{run_id}/results")
def get_run_results(run_id: str, include_images: bool = Query(False)):
    query = supabase.table("analysis_results").select("*").eq("run_id", run_id)
    res = query.execute()
    return {"run_id": run_id, "results": res.data}

@api.get("/results/{result_id}/download-images")
async def download_result_images(result_id: str):
    """
    Download ROI and Mask images for a specific result as a ZIP file.
    """
    logger.info(f"[DOWNLOAD-IMAGES] Fetching images for result {result_id}")
    
    # Get result info
    res = supabase.table("analysis_results").select("filename, roi_image_path, mask_image_path").eq("id", result_id).execute()
    if getattr(res, "error", None) or not res.data:
        raise HTTPException(404, f"Result {result_id} not found")
    
    data = res.data[0]
    filename = data.get("filename", "image")
    roi_url = data.get("roi_image_path")
    mask_url = data.get("mask_image_path")
    
    if not roi_url or not mask_url:
        logger.warning(f"[DOWNLOAD-IMAGES] Historical result {result_id} missing storage paths (ROI: {bool(roi_url)}, Mask: {bool(mask_url)})")
        raise HTTPException(404, "Images not found for this result. This may be an older result processed before image storage was enabled.")
    
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    
    async with httpx.AsyncClient() as client:
        # Download images
        try:
            roi_resp = await client.get(roi_url)
            roi_resp.raise_for_status()
            
            mask_resp = await client.get(mask_url)
            mask_resp.raise_for_status()
        except Exception as e:
            logger.error(f"[DOWNLOAD-IMAGES] Failed to download images from storage: {e}")
            raise HTTPException(500, f"Failed to fetch images from storage: {str(e)}")
        
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Clean filename for the ZIP
            base_name = os.path.splitext(filename)[0]
            zip_file.writestr(f"{base_name}_roi.png", roi_resp.content)
            zip_file.writestr(f"{base_name}_mask.png", mask_resp.content)
    
    zip_buffer.seek(0)
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": f"attachment; filename={base_name}_images.zip"}
    )

# ----------------------------
# Async Job Submission
# ----------------------------

async def process_job(run_id: str, filename: str, file_path: str, analysis_type: str, pixel_size_um: float, 
                      day: Optional[str], organoid_number: Optional[str], day0_area: Optional[float],
                      advanced_params: dict = None, segmentation_method: str = "fiji", 
                      timings: dict = None, total_files: int = None):
    try:
        if timings is None: timings = {}
        
        # Normalize day and organoid_number to strings
        day_str = str(day) if day is not None else "0"
        org_num_str = str(organoid_number) if organoid_number is not None else "1"
        
        logger.info(f"PROCESSING JOB: {filename} | Run: {run_id} | Organoid: {org_num_str} | Day: {day_str}")

        is_day_0 = is_baseline_day(day_str)
        
        is_baseline = is_day_0

        if day0_area is None and not is_day_0 and run_id and org_num_str:
            baseline_options = ["0", "00", "0.0", "0.00", "d00", "d0", "d0.0"]
            res = supabase.table("analysis_results").select("area").eq("run_id", run_id).in_("day", baseline_options).eq("organoid_number", org_num_str).execute()
            logger.info(f"DAY 0 RES: {res.data}")
            if res.data:
                day0_area = res.data[0]["area"]
                logger.info(f"BASELINE FOUND: Using Day 0 area ({day0_area}) for organoid {org_num_str}")

        # Load from disk instead of memory
        with timed(timings, "load_disk_s"):
            def load_img(path):
                return np.array(Image.open(path).convert("RGB"))
            img = await anyio.to_thread.run_sync(load_img, file_path)
        
        params = brightfield_defaults() if analysis_type == "brightfield" else fluorescence_defaults()
        params.update(pixel_size_um=pixel_size_um, return_images=True)
        if advanced_params:
            params.update(advanced_params)
        
        payload = await run_analysis(img, params, timings, "ASYNC", False, day_str, day0_area, org_num_str, analysis_type, segmentation_method=segmentation_method)
        
        # Handle both list and dict formats
        results = payload["results"]
        is_list_format = isinstance(results, list)
        
        if is_list_format:
            # Update growth_rate for all ROIs
            for roi_result in results:
                if is_baseline:
                    roi_result["growth_rate"] = 1.0
                roi_result["growthRate"] = roi_result.get("growth_rate")
            
            # Use largest ROI for logging and baseline calculations
            largest_roi = max(results, key=lambda r: r.get("area", 0))
            gr_val = largest_roi.get("growth_rate")
            area_value = float(largest_roi.get("area", 0))
        else:
            # Backward compatibility: single ROI format
            if is_baseline:
                payload["results"]["growth_rate"] = 1.0
            gr_val = payload["results"].get("growth_rate")
            payload["results"]["growthRate"] = gr_val
            area_value = float(payload["results"].get("area", 0))
        
        row = await persist_one_result(run_id, filename, payload)
        supabase.table("analysis_results").insert(row).execute()
        
        if is_list_format:
            total_area = sum(r.get("area", 0) for r in results)
            logger.info(f"JOB COMPLETED: {filename} | {len(results)} ROIs | Total Area: {total_area:.2f} | Largest Area: {area_value:.2f} | Growth Rate: {gr_val}")
        else:
            logger.info(f"JOB COMPLETED: {filename} | Area: {area_value:.2f} | Growth Rate: {gr_val}")

        # If this was a baseline day, update any other results for the same organoid
        if is_baseline and run_id and org_num_str:
            try:
                logger.info(f"BASELINE DETECTED ({day_str}): Triggering backfill for organoid {org_num_str}")
                
                # Fetch sibling records
                res = supabase.table("analysis_results").select("id, area, day, results_json").eq("run_id", run_id).eq("organoid_number", org_num_str).execute()
                
                updates = 0
                for other in res.data:
                    # Skip the baseline record itself
                    if str(other["day"]) == day_str:
                        continue
                        
                    other_area = float(other["area"])
                    gr = other_area / area_value if area_value > 0 else 1.0
                    
                    # Update column AND JSON
                    rj = other.get("results_json") or {}
                    rj["growth_rate"] = gr
                    rj["growthRate"] = gr
                    
                    supabase.table("analysis_results").update({
                        "growth_rate": gr,
                        "results_json": rj
                    }).eq("id", other["id"]).execute()
                    updates += 1
                    
                if updates > 0:
                    logger.info(f"BACKFILL SUCCESS: Updated {updates} sibling results for organoid {org_num_str}")
            except Exception as e:
                logger.error(f"BACKFILL ERROR: {e}")

        # Check if run is complete (all jobs for this run are finished)
        try:
            # Try to get total from function argument or DB
            total = total_files
            if total is None:
                try:
                    run_res = supabase.table("analysis_runs").select("total_files").eq("id", run_id).single().execute()
                    if run_res.data:
                        total = run_res.data.get("total_files")
                except Exception as e:
                    if "total_files" in str(e):
                        logger.debug("total_files column missing, skipping completion check via DB lookup")
                    else:
                        logger.error(f"Error fetching total_files: {e}")
            
            if total:
                count_res = supabase.table("analysis_results").select("id", count="exact").eq("run_id", run_id).execute()
                current_count = count_res.count or 0
                logger.info(f"Run {run_id} progress: {current_count}/{total}")
                
                if current_count >= total:
                    logger.info(f"Run {run_id} is COMPLETELY FINISHED. Updating status in Supabase.")
                    supabase.table("analysis_runs").update({
                        "status": "completed", 
                        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
                    }).eq("id", run_id).execute()
            else:
                logger.warning(f"Could not determine total_files for run {run_id}. Completion check skipped.")
        except Exception as e:
            logger.error(f"Error during completion check: {e}")

    except Exception as e:
        logger.error(f"Job failed: {filename} in run {run_id}: {e}")
    finally:
        # ALWAYS cleanup the temporary file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"Deleted temp file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete temp file {file_path}: {e}")

# ----------------------------
# Background Worker
# ----------------------------

async def analysis_worker():
    """
    Background worker that processes analysis jobs one by one from the queue.
    This prevents memory exhaustion on low-RAM servers by ensuring sequential processing.
    """
    logger.info("ANALYSIS WORKER: Started and waiting for jobs...")
    while True:
        try:
            # Wait for a job to be added to the queue
            job_data = await analysis_queue.get()
            filename = job_data.get("filename", "unknown")
            run_id = job_data.get("run_id", "unknown")
            
            logger.info(f"ANALYSIS WORKER: Picking up job {filename} (Run: {run_id}) | Queue size: {analysis_queue.qsize()}")
            
            # Execute the job
            # We call the existing process_job function
            await process_job(**job_data)
            
            # Signal that the job is finished
            analysis_queue.task_done()
            logger.info(f"ANALYSIS WORKER: Finished job {filename} | Remaining in queue: {analysis_queue.qsize()}")
            
        except asyncio.CancelledError:
            logger.info("ANALYSIS WORKER: Worker cancelled.")
            break
        except Exception as e:
            logger.exception(f"ANALYSIS WORKER: Unexpected error in worker loop: {e}")
            # Prevent rapid looping if there's a persistent error
            await asyncio.sleep(5)

@api.on_event("startup")
async def startup_event():
    # Start the background worker task
    asyncio.create_task(analysis_worker())

@api.post("/runs/{run_id}/submit-jobs")
async def submit_jobs(
    run_id: str,
    request: Request,
    files: List[UploadFile] = File(...),
    analysis_type: str = Form(...),
    pixel_size_um: float = Form(...),
    segmentation_method: str = Form("fiji"),
    metadata: Optional[str] = Form(None),
    total_files: Optional[int] = Form(None)
):
    meta_list = json.loads(metadata) if metadata else []
    
    # Collect all other form fields as advanced parameters
    form_data = await request.form()
    advanced_params = {}
    known_fields = {"files", "analysis_type", "pixel_size_um", "metadata", "return_images", "segmentation_method", "total_files"}
    
    # Define valid parameter names for filtering
    valid_params = set(brightfield_defaults().keys()) | set(fluorescence_defaults().keys())

    for key, value in form_data.items():
        if key not in known_fields:
            # Map legacy/UI-only parameters
            target_key = key
            if key == "maximum_size":
                target_key = "max_area_px"
            
            if target_key not in valid_params:
                continue

            try:
                # Try to convert to float/int if possible
                if "." in value:
                    advanced_params[target_key] = float(value)
                else:
                    advanced_params[target_key] = int(value)
            except ValueError:
                advanced_params[target_key] = value

    # Create a list of (file, metadata) tuples and sort by day
    # This ensures Day 0 is processed first, which is better for growth rate calculation
    job_items = []
    for i, file in enumerate(files):
        m = meta_list[i] if i < len(meta_list) else {}
        job_items.append((file, m))
    
    def get_day_val(item):
        day = item[1].get("day")
        if is_baseline_day(day):
            return 0
        try:
            import re
            nums = re.findall(r'\d+', str(day))
            if nums:
                return int(nums[0])
            return int(float(day))
        except (ValueError, TypeError):
            return 999 # Put unknown days at the end

    job_items.sort(key=get_day_val)

    total_count = len(files)
    
    async def save_and_enqueue(file_obj, metadata):
        timings = {}
        # Save file to disk instead of keeping data in RAM
        safe_filename = os.path.basename(file_obj.filename)
        temp_filename = f"{run_id}_{uuid4()}_{safe_filename}"
        file_path = os.path.join(UPLOAD_DIR, temp_filename)
        
        try:
            with timed(timings, "upload_save_s"):
                async with await anyio.open_file(file_path, "wb") as buffer:
                    # Use chunks to avoid loading large files into RAM
                    while chunk := await file_obj.read(1024 * 1024): # 1MB chunks
                        await buffer.write(chunk)
            
            await analysis_queue.put({
                "run_id": run_id,
                "filename": file_obj.filename,
                "file_path": file_path,
                "analysis_type": analysis_type,
                "pixel_size_um": pixel_size_um,
                "day": metadata.get("day"),
                "organoid_number": metadata.get("organoid_number"),
                "day0_area": metadata.get("day0_area"),
                "advanced_params": advanced_params,
                "segmentation_method": segmentation_method,
                "timings": timings,
                "total_files": total_files or total_count
            })
        except Exception as e:
            logger.error(f"Failed to save/enqueue file {file_obj.filename}: {e}")
            # Clean up if partially saved
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass

    # Process all file saves in parallel
    await asyncio.gather(*[save_and_enqueue(file, m) for file, m in job_items])
    
    # Update total_files count if the column exists
    try:
        update_data = {"status": "running"}
        if total_files:
            update_data["total_files"] = total_files
            
        supabase.table("analysis_runs").update(update_data).eq("id", run_id).execute()
    except Exception:
        # Fallback if total_files doesn't exist
        supabase.table("analysis_runs").update({"status": "running"}).eq("id", run_id).execute()
        
    return {"run_id": run_id, "job_ids": [str(uuid4()) for _ in files], "status": "submitted"}
