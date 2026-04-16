# Earth Observation using Prithvi EO 2.0

### Crop Analysis & Field Clustering with Foundation Models

This project builds a **complete Earth Observation pipeline** using **Prithvi EO 2.0 embeddings + spectral features + unsupervised learning** to analyze agricultural fields.

It performs:

* Crop zone detection
* NDVI-based vegetation analysis
* Feature extraction using foundation models
* Clustering using PCA + Gaussian Mixture Models (GMM)
* GeoTIFF export + visual dashboards

---

## Key Idea

Instead of relying only on raw spectral indices (like NDVI), this project combines:

* **Prithvi embeddings (deep features)**
* **Spectral indices (NDVI, NDWI, SAVI, NDRE)**
* **Spatial information (pixel coordinates)**

Then clusters field pixels into meaningful crop zones.

---

## Sample Outputs

### Field Analysis Dashboard

<img width="2162" height="1942" alt="prithvi_dashboard_v2_FID991_2024-07-14" src="https://github.com/user-attachments/assets/0605695a-91a5-4517-aa50-a149d579dac5" />




Dashboard shows:

* RGB & NIR views
* NDVI map
* Feature intensity from encoder
* PCA variance + scatter
* Crop cluster map
* Confidence map
* Final interpretation summary

---

## Pipeline Overview

```text
Raw Satellite Chip (6 bands)
        ↓
Spectral Processing (NDVI, NDWI, SAVI, NDRE)
        ↓
Prithvi EO 2.0 Encoder
        ↓
Patch Embeddings → Upsampled to Pixels
        ↓
Feature Fusion:
   [Embeddings + Spectral + Spatial]
        ↓
PCA (Dimensionality Reduction)
        ↓
GMM Clustering (Unsupervised)
        ↓
Crop Zone Mapping + Confidence
        ↓
Dashboard + GeoTIFF Export
```

---

## Project Structure

```bash
.
├── main.py              # Entry point (runs full pipeline)
├── config.py           # Configuration (field ID, paths, parameters)
├── data_loader.py      # Loads chip, mask, metadata
├── spectral.py         # NDVI & spectral indices
├── encoder.py          # Prithvi embedding extraction
├── clustering.py       # PCA + GMM clustering
├── visualization.py    # Dashboard generation
├── export.py           # GeoTIFF export
```

---

## Core Components

### Data Loading

* Loads satellite chip, mask, and metadata
* Ensures proper spatial alignment
   

---

### Spectral Processing

* Computes:

  * NDVI (vegetation health)
  * NDWI (water content)
  * SAVI (soil-adjusted vegetation)
  * NDRE (red-edge proxy)



---

### Prithvi Encoder

* Converts image → transformer embeddings
* Extracts patch tokens
* Upsamples to pixel-level



---

### Clustering (Core Logic)


* Normalize features
* Add spatial coordinates
* Apply PCA (dimensionality reduction)
* Cluster using GMM



Smart handling:

* Small samples → fallback to single cluster
* Confidence estimation using GMM probabilities
* NDVI-based interpretation of clusters

---

### Visualization Dashboard

Generates a **multi-panel analysis dashboard**:

* PCA plots
* NDVI comparison
* Crop zone segmentation
* Confidence heatmaps
* Final interpretation

 

---

### GeoTIFF Export

Exports results for GIS tools:

* Band 1 → cluster labels
* Band 2 → confidence



---

## Example Insights

The model produces interpretations like:

* **Weak spectral separation** → likely stress zones
* **Strong separation** → distinct crop types

Example metrics:

* NDVI gap
* Cluster percentages
* Confidence scores

---

## Configuration

Edit parameters in:

```python
FIELD_ID = 920
DATE = '2024-07-14'

N_CLUSTERS = 2
PCA_COMPONENT = 10
CHIP_SIZE = 224
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

##  Highlight of the project

* Use of **foundation models (Prithvi EO)**
* Combining **deep learning + classical ML (GMM, PCA)**
* Real-world **remote sensing pipeline**
* Strong **visual explainability**
* Export to **GIS-compatible formats**

---

## Future Improvements

* Integrate **temporal time-series analysis**
* Replace GMM with **Deep Clustering / Self-supervised learning**
* Deploy as **web dashboard (Streamlit)**

---

## Acknowledgements

* IBM & NASA – Prithvi EO 2.0
* Open-source geospatial ML ecosystem
* Greenspin GmbH (Würzburg) for providing data, infrastructure, and domain context

---

## Author

**Sudipto Chakraborty**

MSc Aerospace Informatics

University of Würzburg

---

If you like this project,
Give it a ⭐ and feel free to contribute!

