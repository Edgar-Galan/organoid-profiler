# Organoid Profiler

A tool for automated, quantitative characterization of morphological features and fluorescence signals in organoid and spheroid cultures.

- **Web Interface:** [organoid-profiler.com](https://www.organoid-profiler.com/)
    
- **Demo:** [Jupyter Notebook Workflow](https://github.com/Edgar-Galan/organoid-profiler/blob/main/workflow.ipynb)
    

> Citation:
> 
> This code accompanies the manuscript: "Automated, high-throughput and quantitative morphological characterization uncovers conserved longitudinal developmental kinetics in microfluidics-engineered organoids" by Galan et al. (Under review at Nature Communications).
> 
> Full-text preprint available at BioRxiv: [10.64898/2026.01.01.694533v1](https://www.biorxiv.org/content/10.64898/2026.01.01.694533v1).

---

## Overview

Organoid Profiler is an automated software designed to streamline the analysis of microscopy images. It eliminates the need for manual segmentation and metric extraction. By simply uploading your images, Organoid Profiler identifies organoids and extracts 25 comprehensive morphological metrics. This allows researchers to quantitatively discover patterns in data—such as growth rates and phenotypic changes—that are not evident via qualitative visual inspection.

### Core Workflow

1. **Find the Organoid:** Automatically detects organoids in images and generates segmentation masks.
    
2. **Filter Debris:** Applies user-defined criteria (e.g., size or circularity constraints) to ignore dust, shadows, and small cell clumps, preventing data skew.
    
3. **Calculate Metrics:** Computes 25 morphological metrics related to size, shape, pixel intensity, and distribution.
    
4. **Validate Results:** Generates mask overlays on original images, allowing for visual validation of the segmentation accuracy.
    

---

## Installation

You can run the code locally or via the provided [Jupyter Notebook](https://github.com/Edgar-Galan/organoid-profiler/blob/main/workflow.ipynb).

### 1. Create a Conda Environment

Bash

```
# Create the environment
conda create -n organoidprofiler python=3.11 -y

# Activate the environment
conda activate organoidprofiler
```

### 2. Install Scientific Packages

Bash

```
# Install scientific packages with conda
conda install -c conda-forge numpy scipy scikit-image matplotlib pillow -y
```

### 3. Install Dependencies via Pip

Bash

```
# Install the remaining dependencies
pip install cellpose==2.0.2 torch fastapi uvicorn pydantic-settings supabase loguru python-dotenv httpx python-multipart anyio
```

### 4. (Optional) Jupyter Notebook Setup

If you plan to run the workflow via the notebook:

Bash

```
# Install the kernel
pip install ipykernel numpy

# Register the kernel
python -m ipykernel install --user --name=organoidprofiler --display-name "Org Profiler"
```

---

## Usage Guide

### 1. Choose Analysis Mode

The tool offers two segmentation and analysis modes tailored to different datasets:

|**Mode**|**Target Images**|**Primary Use Cases**|**Key Features**|
|---|---|---|---|
|**Brightfield (Morphology)**|Standard light microscopy|Measuring size, shape, swelling, and disintegration (death).|Tracks individual organoids across time-points to calculate specific growth rates.|
|**Fluorescence (Intensity)**|Immunofluorescence, Live/Dead staining|Gene expression, protein levels, cell viability.|Ignores saturated pixels; performs automated background subtraction for standardized Corrected Total Fluorescence (CTF).|

### 2. Prepare and Upload Images

- **Supported Formats:** `.jpg`, `.png`, `.tif`, `.tiff` (Multichannel support, e.g., `.nd2`, coming soon).
    
- File Naming Convention:
    
    To enable time-point tracking, organize your folders or name your files as follows:
    
    - **Option A (Filename):** `experiment-name_d[dd]_org[oo].jpg` or `experiment-name_d[dd]_[oo].jpg`
        
        - `[dd]`: Day of experiment.
            
        - `[oo]`: (Optional) Organoid identifier number for individual longitudinal tracking.
            
    - **Option B (Folder Structure):** `experiment-name/d[dd]/image.jpg` or `experiment-name/Day[dd]/image.jpg`
        
    
    > _Note: Example datasets with this naming convention can be found in the [dataset](https://github.com/Edgar-Galan/organoid-profiler/tree/main/dataset) folder of this repository._
    

### 3. Interpret Results

The tool outputs a data table in **`.csv`** format containing the 25 extracted metrics.

- **Coming Soon:** Automated plotting (line plots, strip plots, clustermaps, variance heatmaps) and AI-integrated reporting to generate scientific descriptions of data patterns.
    

### 4. Quality Control (Visual Check)

For every processed image, the tool saves a **ROI Overlay** image.

- **Check for:** A **magenta line** outlining the organoid.
    
- **Action:**
    
    - Line follows contour $\rightarrow$ **Valid Data.**
        
    - Poor segmentation $\rightarrow$ Adjust parameters (Area/Circularity thresholds for standard mode; Flow/Probability thresholds for deep learning mode).
        

---

## Metrics Dictionary

The following 25 metrics are calculated for every identified object:

|**Feature**|**Description**|**Equation**|**Notes**|
|---|---|---|---|
|**Area**|Projected area of the object (ROI).|$A$|Calibrated in $\mu m^2$ based on `pixel_size`.|
|**Growth Rate**|Relative change in area normalized to $t_0$.|$G_r = \frac{A_d}{\bar{A}_{t0}}$|$A_d$: current area; $\bar{A}_{t0}$: mean area on day 0.|
|**Perimeter**|Length of the outside boundary.|$P$|Calculated from boundary pixel centers.|
|**Feret Max**|Longest distance between any two points on boundary.|$F_{max}$|Max caliper diameter.|
|**Feret Min**|Min distance between parallel tangents.|$F_{min}$|Min caliper diameter.|
|**Major Axis**|Primary axis of best-fitting ellipse.|$Major$||
|**Minor Axis**|Secondary axis of best-fitting ellipse.|$Minor$||
|**Aspect Ratio**|Ratio of major to minor axis.|$AR = \frac{Major}{Minor}$|1.0 = Circle; >1.0 = Elongated.|
|**Equiv. Diameter**|Diameter of a circle with same area.|$ECD = 2 \sqrt{\frac{A}{2\pi}}$||
|**Circularity**|Resemblance to a perfect circle.|$C = 4\pi \times \frac{A}{P^2}$|1.0 = Perfect circle; $\to$ 0.0 = Elongated polygon.|
|**Roundness**|Inverse AR derived from area.|$R = 4 \times \frac{A}{\pi \times Major^2}$|Insensitive to irregular borders (smoothness).|
|**Solidity**|Density relative to convex hull.|$S = \frac{A}{A_{convex}}$|Measures convexity.|
|**Mean Intensity**|Average pixel intensity.|$\bar{I} = \frac{\sum I_{xy}}{N}$|$N$ = number of pixels.|
|**Int. Density**|Sum of pixel intensities.|$IntDen = \bar{I} \times A$||
|**Corr. Total Fluor.**|Total fluorescence corrected for background.|$CTF = IntDen - (A \times \bar{I}_{bg})$|$\bar{I}_{bg}$ = Background mean intensity.|
|**Corr. Mean Fluor.**|Mean intensity corrected for background.|$CMF = \frac{CTF}{A}$|Average signal density per unit area.|
|**Skewness**|Asymmetry of intensity distribution.|$Skew = \frac{1}{N} \sum (\frac{x_i - \bar{x}}{\sigma})^3$||
|**Kurtosis**|"Tailedness" of intensity distribution.|$Kurt = [\dots]^4 - 3$|Flatness of pixel value distribution.|
|**Centroid Dist.**|Geometric vs. Intensity-weighted center shift.|$d = \sqrt{(x - x_m)^2 + (y - y_m)^2}$|Indicates uneven density.|
|**Eccentricity**|Deviation of fitted ellipse from circle.|$e = \sqrt{1 - \frac{Minor^2}{Major^2}}$|0 = Circular; $\to$ 1 = Elliptical.|
|**Criteria**|Boolean validation of object.|$Valid = (A > S_{min}) \land (C > C_{thresh})$|Used to filter artifacts.|

---

## Contact & Support

If you encounter issues or the tool does not fulfill your experimental needs, please contact us. We appreciate your feedback.

**Email:** [edgar.galan@tsinghua.org.cn](mailto:edgar.galan@tsinghua.org.cn)
