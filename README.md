# Organoid Profiler
A tool for the automated, quantitative characterization of morphological features and fluorescence signals of organoid and spheroid cultures.

A web interface for this project is available at https://organoid-profiler.com/

This code accompanies the manuscript entitled:
"Automated, high-throughput and quantitative morphological characterization uncovers conserved longitudinal developmental kinetics in microfluidics-engineered organoids" by Galan et al. Under review at Nature Communications. 

# **Welcome to Organoid Profiler**

## **Installation instructions**

These are the installation instructions to be able to run the code and the jupyter notebook. In this repo, we have included the server that is available with a easy-to-use user interface on this [website](https://organoid-profiler.com/) . 

### Create the Conda Environment

```bash
# create the environment
conda create -n orgprofiler python=3.11 -y

# activate the environment
conda activate orgprofiler
```

### Use Conda to Install Scientific Packages

```bash
# install scientific packages with conda
conda install -c conda-forge numpy scipy scikit-image matplotlib pillow -y
```

### Install All Other Packages Using pip

```bash
# install the remaining dependencies with pip
pip install cellpose torch fastapi uvicorn pydantic-settings supabase loguru python-dotenv httpx
```

### (Optional) Jupyter Notebook Setup

If you are planning on running the Jupyter notebook, run the following commands:

```bash
# install the kernel
pip install ipykernel numpy

# register the kernel; you can then select "orgprofiler" environment to run the Jupyter notebook
python -m ipykernel install --user --name=orgprofiler --display-name "Org Profiler"
```

## **What is this tool?**

This is an automated software designed to analyze microscopy images of organoids. It replaces the need for manual tracing in ImageJ/Fiji. Instead of drawing circles by hand, you upload your images, and the software automatically identifies the organoid, measures it, and calculates growth or fluorescence intensity.

## **What does it do?**

1. **Finds the Organoid:** Automatically detects the organoid in your image and draws an outline (Region of Interest or ROI) around it.
    
2. **Ignores Debris:** Filters out dust, shadows, and small cell clumps so they don't skew your data.
    
3. **Calculates Metrics:** Instantly computes Area, Diameter, Circularity, and Signal Intensity.
    
4. **Validates Data:** Generates a "check image" with a magenta outline so you can visually confirm the software measured the right thing.
    

---

## **How to Use It**

### **1. Choose Your Analysis Mode**

The tool has two specific modes depending on your microscope settings.

- **Brightfield Mode (Morphology)**
    
    - **What it's for:** Standard light microscopy where organoids appear **dark** against a **bright** background.
        
    - **Use this to measure:** Size, growth, swelling, or death (disintegration).
        
    - **Key Feature:** If you provide the starting area (Day 0 area), it calculates a **Growth Rate** (e.g., 1.2x growth) automatically.
        
- **Fluorescence Mode (Intensity)**
    
    - **What it's for:** Fluorescent images (GFP, RFP, DAPI, etc.) where organoids appear **bright** against a **dark** background.
        
    - **Use this to measure:** Gene expression, protein levels, or cell viability.
        
    - **Key Feature:** It automatically subtracts background noise to provide **Corrected Total Fluorescence (CTF)**, the standard metric for accurate brightness comparison.
        

### **2. Upload Your Images**

- **Supported Files:** `.png`, `.jpg`, `.tif`, `.bmp`, `.gif`.
    
- **Best Practices:**
    
    - Ensure there is only **one organoid per image** for the most accurate results.
        
    - Ensure the organoid is **not touching the edge** of the image (the software automatically rejects edge-touching objects to prevent partial measurements).
        

### **3. Interpret Your Results**

The tool outputs a data table. Here is how to read the most important columns:

|**Metric**|**What it tells you**|
|---|---|
|**Area**|The total surface area of the organoid (in $\mu m^2$).|
|**Growth Rate**|How much the organoid grew relative to Day 0. (e.g., `1.5` means it is 50% larger).|
|**Circularity**|A shape score from 0.0 to 1.0. A perfect circle is `1.0`. Lower values (e.g., `0.5`) indicate an irregular, budded, or disintegrating shape.|
|**Feret Diameter**|The "caliper" width. Imagine measuring the organoid at its widest point with a ruler. Useful for oblong shapes.|
|**Corrected Total Fluorescence**|The total brightness of the organoid with background noise removed. Use this to compare signal strength between samples.|

### **4. Quality Control (Visual Check)**

For every image, the tool saves a **ROI Overlay** image.

- **Look for:** A **magenta line** outlining your organoid.
    
- **Action:** Open these images to verify accuracy.
    
    - If the line hugs the organoid perfectly $\rightarrow$ **Keep the data.**
        
    - If the line circles debris or misses part of the organoid $\rightarrow$ **Discard that data point.**
        

---

## **Troubleshooting**

- **My organoid wasn't measured (Result is "NA").**
    
    - Is it touching the edge of the image? (The tool ignores these).
        
    - Is it extremely small? (It might be below the minimum size filter).
        
- **My fluorescence value is negative.**
    
    - This occurs if the background is brighter than the object (e.g., high noise, no signal). It indicates no significant fluorescence was detected.

# Technical details

**1. Overview**

This system provides an automated pipeline for analyzing microscopy images of organoids. It is designed using a robust, Python-based FastAPI service. The system supports two distinct imaging modalities:

- **Brightfield (BF):** For analyzing organoid morphology (size, shape, growth, etc).
    
- **Fluorescence (FL):** For analyzing signal intensity and area.
    

The core output includes quantitative metrics (Area, Feret diameter, Circularity, Corrected Total Fluorescence) and visual validation images (ROI overlays).

---

## **2. Image Processing Logic**

The analysis logic is encapsulated in `main.py` within the `analyze_image` function. The process follows a linear pipeline of **Segmentation $\rightarrow$ Filtering $\rightarrow$ Measurement**.

### **Step A: Pre-processing**

1. **Input:** Accepts an image file (PNG, JPG, TIF).
    
2. **Conversion:** Decodes the image into a NumPy array (RGB).
    
3. **Cropping (Optional):** Removes a configurable border (`crop_border_px`) from the image edges to eliminate camera artifacts or frame lines often found in microscope exports.
    

### **Step B: Segmentation (Mask Generation)**

The system generates a binary mask (Foreground vs. Background) using a "Fiji-style" morphological approach.

1. **Grayscale Conversion:** The RGB image is converted to 8-bit grayscale using standard luminance weights ($0.299R + 0.587G + 0.114B$).
    
2. **Initial Thresholding (Isodata):**
    
    - **BF Mode:** Assumes **dark** objects. Pixels $\le$ Threshold are foreground.
        
    - **FL Mode:** Assumes **bright** objects. Pixels $\ge$ Threshold are foreground.
        
3. **Morphological Cleaning:**
    
    - **Hole Filling:** Fills internal holes in the detected objects.
        
    - **Dilation:** Expands the mask by `dilate_iter` pixels. This connects fragmented parts of an organoid.
        
    - **Erosion:** Shrinks the mask by `erode_iter` pixels. This restores the object to its approximate original size while smoothing rough edges.
        
4. **Gaussian Smoothing:** A Gaussian blur (`sigma_pre`) is applied to the binary mask itself to create smooth, organic contours rather than jagged pixelated edges.
    
5. **Secondary Thresholding:** The smoothed mask is thresholded again to finalize the binary ROI (Region of Interest).
    

### **Step C: Object Filtering & Selection**

The system identifies connected components (contours) in the mask and filters them to find the true organoid.

1. **Area Filter:** Rejects objects smaller than `min_area_px` or larger than `max_area_px`.
    
2. **Circularity Filter:** Rejects objects with circularity below `min_circ` (useful for ignoring debris in Brightfield).
    
3. **Edge Exclusion:** (Optional) Rejects objects touching or near the image border (`edge_margin`).
    
4. **Selection Strategy:**
    
    - **BF Strategy (`"largest"`):** Selects the single largest valid object.
        
    - **FL Strategy (`"composite_filtered"`):** Combines _all_ valid objects into one composite ROI. This captures organoids that may appear as disjointed bright spots.
        

### **Step D: Measurements**

Once the ROI is finalized, metrics are calculated.

#### **1. Morphological Metrics**

- **Area:** Total area in $\mu m^2$ (converted using `pixel_size_um`).
    
- **Perimeter:** Length of the contour.
    
- **Circularity:** $4\pi \times (Area / Perimeter^2)$. A value of 1.0 is a perfect circle.
    
- **Feret Diameter:** The "caliper" dimensions (Maximum and Minimum caliper width) calculated from the convex hull of the ROI.
    
- **Fitted Ellipse:** Major axis, Minor axis, and Aspect Ratio.
    

#### **2. Intensity Metrics**

- **Inversion (BF Only):** Brightfield images are inverted ($255 - pixel\_value$) so that dark organoids yield high "intensity" values, allowing for density calculations.
    
- **Raw Metrics:** Mean, Median, Mode, Min, Max, Standard Deviation, and Integrated Density (Sum of all pixels).
    
- **Skewness & Kurtosis:** Statistical descriptors of the intensity distribution.
    

#### **3. Background Correction**

To calculate accurate fluorescence or density, background noise is subtracted.

- **Ring Method (BF Default):** Measures the median intensity of a ring surrounding the organoid (width `ring_px`).
    
- **Inverse Composite (FL Default):** Measures the median intensity of _all_ pixels outside the detected objects.
    

Key Formula: Corrected Total Fluorescence (CTF)

$$CTF = IntegratedDensity - (Area_{ROI} \times Mean_{Background})$$

### **Step E: Growth Rate**

If `day` and `day0_area` are provided in the request:

- **Day 0:** Growth Rate = 1.0
    
- **Day N:** Growth Rate = $Area_{DayN} / Area_{Day0}$
    

---

## **3. API Reference**

### **POST** `/analyze/brightfield`

Optimized for dark objects on light backgrounds.

**Key Parameters:**

- `pixel_size_um`: (Float) Calibration scale (microns per pixel).
    
- `min_area_px`: (Int) Minimum size to detect (Default: 60,000).
    
- `min_circ`: (Float) Minimum circularity (Default: 0.28).
    
- `day0_area`: (Float, Optional) The area of this organoid on Day 0 (for growth calculation).
    

### **POST** `/analyze/fluorescence`

Optimized for bright signal on dark backgrounds.

**Key Parameters:**

- `pixel_size_um`: (Float) Calibration scale.
    
- `sigma_pre`: (Float) Higher smoothing (Default: 14.0) to merge "spotty" fluorescence signal.
    
- `select_strategy`: Defaults to "composite_filtered" to measure total signal area.
    

**Returns (JSON):**

JSON

```
{
  "results": {
    "area": 150000.5,
    "circ": 0.85,
    "mean": 45.2,
    "corrTotalInt": 500200.0,
    "feret": 450.2,
    "growthRate": 1.25,
    …
  },
  "roi_image": "data:image/png;base64,…",  // Overlay image
  "mask_image": "data:image/png;base64,…"   // Binary mask
}
```

---

## **4. Database Integration (Supabase)**

The system includes endpoints to persist results to a Supabase database.

1. **Create Run:** `POST /runs` - Creates a tracking ID for a batch of images.
    
2. **Persist Results:** `POST /runs/{run_id}/persist`
    
    - Uploads the generated ROI and Mask images to Supabase Storage.
        
    - Saves all calculated metrics (JSON output) into the `analysis_results` SQL table.
        
    - Links results to the specific `run_id`.
        

## **5. Configuration & Requirements**

- **Environment Variables:**
    
    - `SUPABASE_URL`: URL of the Supabase instance.
        
    - `SUPABASE_KEY`: Service role or anon key.
        
- **Dependencies:** `fastapi`, `numpy`, `scikit-image`, `scipy`, `pillow`, `supabase`, `uvicorn`.
    
- **Optional:** `cellpose` (if deep-learning segmentation is enabled).
