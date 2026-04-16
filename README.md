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

## 📸 Sample Outputs

### Field Analysis Dashboard

![Image](https://images.openai.com/static-rsc-4/ywU-glNz8JKPKTu-NlHq4LOV6x9NrNuyLwt2eMWacxZwyxMhDMQ7tVzE7tmEsChM8WC6gwrbzPH5J3u9sxStk_Qw8rOnChv93gbUnqozZMlDYCJoKSGaPFdFlhkKQtceTO2V0Wa74411tUJa3gsNEuve4mpwCJttLXFWB_nKNeSNswUKPLN1mBy6nkXOtwOz?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/HOwD3L5kShLQzmXJU8HRFpcaQkD9HJibUomEpjPgTatOV04P-qZr2-nXgLAMR4y4wp1mL1EIzlk0tECwWeH6owsYHoBxZ64vgpHPM2HTg35KDrGuMDJpecZ9hDIwUdl1Fh-G2VPafpc-r_dlPw2AiwXz6C5LuF_19UWbb148Jqytgv6M823n_4mhG1UfVT_a?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/Wo7ZZT1glk5csri8DFkPgHEGznwDcSjGsrGlDudIy1MLm9_d680z2KO-qghjAXasH-I7Irb8zARS9IpZp2ZScabVDZS9vPjSr94tHl1WKmsyqWcUOZCxjwJQ_dqNUCG_m8uyTlisZHZNTL0NhKZFMM9IjAYv-5qdb4OSswAWmj7HyTxQfaVlQxHRG0WVwWLy?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/MO7o1p5SiLzDLN0C7qlFb5ZLMjd0KNjwbs_SjF6MY-lB8c5DfR-zfi8hQa5DyuJrkXcrUDdscBEeslUPgsDczRAxYCCnI9cr9O04tsR1Jju6dYj5JjGTkXrp-KD3C4-EO8ViOdtqN8L1TUDyl_QzrD5iupNcxD4EsxhG4YmKAqPViBCrEb7j1cstYDA3ciSf?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/DZUkCt2oesW4qDGLZUzmZrESTb0et1nc4nzEay0UqUHkhaYfADhgLSkzdcczluKK8DQSUkdh2o6EARw6HWanHLKKfNaboq0FqmXgzNnMWt7DISVnPPg7bVXhSG5S2__iL8zkWqThGPw_937Vm7gOr_xsRSdqubi6ODohuYToaNpsaq1yT0AA4DTKqFV4Ujxi?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/D8X3UHQtWBflVwkURflYOYOjFywDn976kvrrXPapUkHhJNBXyNH0FgtFNLLeWPOxnjW3Umbj9Xb4ko00ffu793RJ-bB8oPhZWn2MVIZIwAHLYXlJ7x2CiV_S8CECedZtgf8selFXQ4iz8w3MIEyOAGokFRpW_hHxs3R5UG5_VcF4qJiLNmopZvrQpvv4RtVn?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/4rXoM8Pk5oWYsdza_MJWhCVR-YPP4K-9D9KXCBvuFapzVqgTEUAEUiR5MvhdUZsNFXjqr2tA30z3_asoMQuB2ayN-TKMgrOvgEX3junV6HtlegdjoLIuDBTXzDKN_AnZovKPvoOSXzVmFTHptNUKlGpWT4iARQjCMnpKTP7KYtQvNEsr3J8k495hugwbiLvL?purpose=fullsize)

Each dashboard shows:

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

👉 Implemented in: 

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

👉 

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

Use of **foundation models (Prithvi EO)**
Combining **deep learning + classical ML (GMM, PCA)**
Real-world **remote sensing pipeline**
Strong **visual explainability**
Export to **GIS-compatible formats**

---

## Future Improvements

* Integrate **temporal time-series analysis**
* Replace GMM with **Deep Clustering / Self-supervised learning**
* Deploy as **web dashboard (Streamlit)**

---

## Acknowledgements

* IBM & NASA – Prithvi EO 2.0
* Open-source geospatial ML ecosystem

---

## Author

**Sudipto Chakraborty**
MSc Aerospace Informatics
University of Würzburg

---

If you like this project,
Give it a ⭐ and feel free to contribute!

