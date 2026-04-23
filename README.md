# Earth Observation using Prithvi EO 2.0

### Temporal Crop Analysis & Multi-Cropping Detection with Foundation Models

This project builds a **full temporal Earth Observation pipeline** using **Prithvi EO 2.0 + spectral features + temporal statistics + unsupervised learning** to analyze **intra-field crop variability (multi-cropping / stress zones)**.

---

## (Latest Version)

Temporal stack (multi-date satellite data)
Cloud & shadow masking (SCL + spectral fusion)
Temporal NDVI + embedding statistics
Automatic cluster selection using **BIC**
Phenotype-based field analysis (instead of simple clusters)
Temporal NDVI trajectory visualization
Strong interpretability + confidence estimation

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

---

## Sample Output (Temporal Dashboard)

<img width="2162" height="1942" alt="prithvi_dashboard_v2_FID991_2024-07-14" src="C:\Users\Sudipto\internship\EO\prithvi\greenspin\multi_crop_output\FID_2701\prithvi_dashboard_v2_Temporal.png">

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

## 🧠 Pipeline Overview

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
Temporal Analysis + Visualization
        ↓
Dashboard + GeoTIFF Export
```

---

## Project Structure

```bash
.
├── main.py                # Full temporal pipeline execution :contentReference[oaicite:0]{index=0}
├── config.py             # Field + temporal configuration (multi-date support) :contentReference[oaicite:1]{index=1}
├── data_loader.py        # Loads temporal chips, masks, metadata :contentReference[oaicite:2]{index=2}
├── cloud_mask.py         # SCL + spectral cloud masking :contentReference[oaicite:3]{index=3}
├── spectral.py           # NDVI, indices, temporal composite :contentReference[oaicite:4]{index=4}
├── encoder.py            # Prithvi embedding + temporal stats :contentReference[oaicite:5]{index=5}
├── clustering.py         # PCA + GMM + BIC + validation :contentReference[oaicite:6]{index=6}
├── visualization.py      # Multi-panel temporal dashboard :contentReference[oaicite:7]{index=7}
├── export.py             # GeoTIFF export :contentReference[oaicite:8]{index=8}
├── modelfactory.py       # Prithvi model loading (local weights)
├── qgis_chip_extractor.py# Data extraction from Sentinel-2 (QGIS) :contentReference[oaicite:9]{index=9}
```

---

## Core Components

### Cloud & Shadow Masking

* Combines:

  * Sentinel-2 SCL labels
  * Spectral thresholding
* Ensures only **clean pixels are used**
* Robust fallback when SCL is unavailable 

---

### Spectral Processing

* Computes:

  * NDVI (vegetation)
  * NDWI (water)
  * SAVI (soil-adjusted)
  * NDRE (red-edge proxy) 

---

### Prithvi EO Encoder

* Processes **temporal satellite stacks**
* Extracts:

  * Patch embeddings (per date)
  * Temporal statistics (mean, std, range)



---

### Temporal Feature Engineering

* NDVI trajectories per pixel
* Growth patterns across season
* Embedding variability over time



---

### Clustering (Core Logic)

* Feature fusion:

  ```
  Embeddings + Temporal + Spectral + Spatial
  ```
* PCA for dimensionality reduction
* **GMM clustering with automatic BIC selection**
* Quality metrics:

  * Silhouette score
  * Davies-Bouldin index

Fully adaptive clustering pipeline 

---

### Phenotype Mapping

Instead of raw clusters → meaningful **phenotypes**:

* High NDVI → healthy crop zones
* Low NDVI → stress / weak growth
* Mixed → possible multi-cropping

---

### Temporal Analysis

Tracks:

* NDVI evolution over time
* Growth differences between zones

Key insight:

> Same field ≠ same behavior over time

---

### GeoTIFF Export

* Band 1 → phenotype labels
* Band 2 → confidence

→ Ready for GIS tools (QGIS, ArcGIS) 

---

## Example Insights

The system can detect:

* **Multi-cropping within a field**
* **Stress zones vs healthy regions**
* **Growth differences over time**
* **Weak vs strong spectral separation**

Example outputs:

* NDVI gap between phenotypes
* Cluster distribution (%)
* Confidence scores
* Temporal NDVI curves

---

## Configuration

```python
FIELD_ID = 2701

DATES = [...]   # 37 temporal observations

MAX_CLUSTERS = 8
PCA_COMPONENT = 10

CHIP_SIZE = 224
```

---

## ▶️ How to Run

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

* Foundation model usage (Prithvi EO 2.0)
* Temporal + spatial + spectral fusion
* Fully unsupervised learning pipeline
* Strong interpretability 
* Real-world agricultural application
* Works on Sentinel-2 satellite data
* Temporal crop behavior modeling

---

## 🔮 Future Improvements

* Supervised crop classification (if labels available)
* Multimodal fusion (weather, soil data)
* Deep clustering / self-supervised learning
* Real-time monitoring system
* Web deployment (Streamlit / dashboard)

---

## Acknowledgements

* IBM & NASA – Prithvi EO 2.0
* Open-source geospatial ML ecosystem
* Greenspin GmbH (Würzburg) for providing data, infrastructure, and domain support

---

## Author

**Sudipto Chakraborty**
MSc Aerospace Informatics
University of Würzburg

---

If you like this project,
Give it a ⭐ and feel free to contribute!


