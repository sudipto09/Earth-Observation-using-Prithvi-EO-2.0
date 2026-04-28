````markdown
# Earth Observation using Prithvi EO 2.0

### Temporal Crop Analysis & Multi-Cropping Detection with Foundation Models

This project builds a **full temporal Earth Observation pipeline** using **Prithvi EO 2.0 + spectral features + temporal statistics + unsupervised learning** to analyze **intra-field crop variability (multi-cropping / stress zones)**.

The main objective is to detect and understand **multi-cropping patterns within agricultural fields**, identify crop variability, and analyze field-level growth behavior across time using satellite imagery and foundation model embeddings.

---

## (Latest Version)

Temporal stack (multi-date satellite data)

Cloud & shadow masking (SCL + spectral fusion)

Temporal NDVI + embedding statistics

Automatic cluster selection using **BIC**

Phenotype-based field analysis (instead of simple clusters)

Temporal NDVI trajectory visualization

Per-date clustering analysis

Strong interpretability + confidence estimation

GeoTIFF export for GIS workflows

---

## Key Idea

Instead of analyzing a **single snapshot**, this project uses:

* **Prithvi EO embeddings (deep features across time)**
* **Spectral indices (NDVI, NDWI, SAVI, NDRE)**
* **Temporal statistics (growth patterns, variability)**
* **Spatial information**

to identify **hidden patterns inside fields**, such as:

* Multiple crops (multi-cropping)
* Growth differences
* Stress zones
* Temporal crop behavior changes

---

## Sample Output (Temporal Dashboard)

![Temporal Dashboard](prithvi/greenspin/multi_crop_output/FID_2701/prithvi_dashboard_v2_Temporal.png)

### Dashboard shows:

* RGB & NIR views (best cloud-free date)
* NDVI map (vegetation health)
* Encoder feature intensity
* BIC-based cluster selection
* PCA feature space
* Phenotype map (crop zones)
* Confidence map
* Mean NDVI per phenotype
* Temporal NDVI trajectories (37 dates)
* Final field summary

---

## Pipeline Overview

```text
Multi-Date Satellite Data (T × 6 bands)
        ↓
Cloud & Shadow Masking (SCL + Spectral)
        ↓
Temporal NDVI Computation
        ↓
Temporal Composite (best clear pixels)
        ↓
Prithvi EO 2.0 Encoder
        ↓
Patch Tokens (per date)
        ↓
Temporal Embedding Statistics
        ↓
Upsampling → Pixel Features
        ↓
Feature Fusion:
   [Embeddings + Temporal + Spectral + Spatial]
        ↓
PCA (Dimensionality Reduction)
        ↓
GMM Clustering (with BIC selection)
        ↓
Phenotype Mapping + Confidence
        ↓
Per-Date Clustering Validation
        ↓
Temporal Analysis + Visualization
        ↓
Dashboard + GeoTIFF Export
````

---

## Project Structure

```bash
.
├── main.py                    # Full temporal pipeline execution
├── config.py                  # Field + temporal configuration
├── data_loader.py             # Loads temporal chips, masks, metadata
├── cloud_mask.py              # SCL + spectral cloud masking
├── spectral.py                # NDVI, indices, temporal composite
├── encoder.py                 # Prithvi embedding + temporal statistics
├── clustering.py              # PCA + GMM + BIC + validation
├── per_date_clustering.py     # Per-date clustering consistency analysis
├── visualization.py           # Multi-panel temporal dashboard
├── export.py                  # GeoTIFF export
├── modelfactory.py            # Prithvi model loading (local weights)
├── qgis_chip_extractor.py     # Sentinel-2 chip extraction from QGIS
```

---

## Core Components

### Cloud & Shadow Masking

* Combines:

  * Sentinel-2 SCL labels
  * Spectral thresholding

* Ensures only **clean pixels are used**

* Robust fallback when SCL is unavailable

This improves temporal consistency and prevents cloud contamination from affecting clustering results.

---

### Spectral Processing

* Computes:

  * NDVI (vegetation)
  * NDWI (water)
  * SAVI (soil-adjusted vegetation)
  * NDRE (red-edge proxy)

* Builds temporal composite imagery using the **greenest cloud-free pixels**

This helps preserve the most informative vegetation signals across the season.

---

### Prithvi EO Encoder

* Processes **multi-temporal satellite stacks**

* Extracts:

  * Patch embeddings (per date)
  * Temporal embedding statistics:

    * mean
    * standard deviation
    * temporal range

These embeddings serve as the deep feature backbone of the project.

---

### Temporal Feature Engineering

* NDVI trajectories per pixel
* Growth patterns across the season
* Embedding variability over time
* Temporal stability analysis

This helps distinguish crop behavior beyond what a single image can show.

---

### Clustering (Core Logic)

* Feature fusion:

```text
Embeddings + Temporal + Spectral + Spatial
```

* PCA for dimensionality reduction

* **GMM clustering with automatic BIC selection**

* Quality metrics:

  * Silhouette score
  * Davies-Bouldin index
  * Cluster confidence estimation

This creates a fully adaptive and robust clustering pipeline.

---

### Phenotype Mapping

Instead of raw clusters → meaningful **phenotypes**

Examples:

* High NDVI → healthy crop zones
* Low NDVI → stress / weak growth
* Mixed patterns → possible multi-cropping
* Temporal instability → abnormal growth behavior

This improves interpretability for real agricultural decision-making.

---

### Per-Date Clustering Analysis

The system also performs clustering across individual dates to validate:

* temporal consistency
* cluster stability
* seasonal crop transitions

This helps verify whether patterns remain stable or change significantly over time.

---

### Temporal Analysis

Tracks:

* NDVI evolution over time
* Growth differences between zones
* Phenotype-specific crop trajectories

Key insight:

> Same field ≠ same behavior over time

This is one of the strongest research contributions of the project.

---

### GeoTIFF Export

Exports:

* Band 1 → phenotype labels
* Band 2 → confidence values

Ready for:

* QGIS
* ArcGIS
* GIS-based agricultural workflows

---

## Example Insights

The system can detect:

* **Multi-cropping within a field**
* **Stress zones vs healthy regions**
* **Growth differences over time**
* **Weak vs strong spectral separation**
* **Phenotype consistency across dates**

Example outputs:

* NDVI gap between phenotypes
* Cluster distribution (%)
* Confidence scores
* Temporal NDVI curves
* Seasonal growth comparisons

---

## Configuration

```python
FIELD_ID = 2701

DATES = [...]   # 37 temporal observations

MAX_CLUSTERS = 8
PCA_COMPONENT = 10

CHIP_SIZE = 224
PATCH_GRID = 14
```

---

## How to Run

### 1. Install dependencies

```bash
pip install numpy torch scikit-learn matplotlib rasterio
```

### 2. Run pipeline

```bash
python main.py
```

---

## Highlights

* Foundation model usage (**Prithvi EO 2.0**)
* Temporal + spatial + spectral feature fusion
* Fully unsupervised learning pipeline
* Automatic cluster count selection using BIC
* Strong interpretability and explainability
* Real-world agricultural application
* Works on Sentinel-2 satellite imagery
* Temporal crop behavior modeling
* GIS-compatible outputs
* Multi-cropping detection pipeline

---

## Future Improvements

* Supervised crop classification (if labels become available)
* Multimodal fusion (weather + soil + sensor data)
* Deep clustering / self-supervised learning
* Real-time monitoring system
* Streamlit dashboard deployment
* Integration with precision agriculture decision systems

---

## Acknowledgements

* IBM & NASA – Prithvi EO 2.0
* Open-source geospatial ML ecosystem
* Greenspin GmbH (Würzburg) for providing data, infrastructure, imagery, and domain support during the internship

---

## Author

**Sudipto Chakraborty**

MSc Aerospace Informatics

University of Würzburg

---

If you like this project,
give it a ⭐ and feel free to contribute!

```
```
