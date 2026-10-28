# 🌾 GIS Rice Land Suitability Analysis — Nigeria

A Python-based GIS tool for identifying suitable lands for rice cultivation across Nigeria using **Multi-Criteria Decision Analysis (MCDA)** and the **Analytic Hierarchy Process (AHP)** for weight derivation.

---

## 📌 Overview

This project performs a spatial land suitability analysis for rice farming in Nigeria by integrating six key environmental and socioeconomic criteria:

| Criterion | Weight (AHP) | Description |
|---|---|---|
| **Soil** | 34.2% | pH, organic carbon, clay/sand content, bulk density |
| **Climate** | 20.6% | Temperature, precipitation, seasonality (WorldClim BIO variables) |
| **Water** | 17.0% | Proximity to water bodies and irrigation potential |
| **Land Cover** | 13.4% | Existing land use compatibility |
| **Topography** | 8.5% | Slope, elevation, and aspect |
| **Socioeconomic** | 6.3% | Market access and infrastructure proximity |

Criteria scores are aggregated using a **weighted overlay**, and the final suitability map is classified into four FAO-standard classes:

- **S1** — Highly Suitable (score ≥ 0.75)
- **S2** — Moderately Suitable (score ≥ 0.50)
- **S3** — Marginally Suitable (score ≥ 0.25)
- **N** — Not Suitable (score < 0.25)

---

## 🗂 Project Structure

```
GIS land use/
├── src/
│   ├── main.py              # CLI entry point
│   ├── ahp/
│   │   ├── pairwise.py      # Pairwise comparison matrix
│   │   └── weights.py       # AHP weight computation & consistency check
│   ├── criteria/
│   │   ├── climate.py       # Climate scoring (WorldClim BIO variables)
│   │   ├── soil.py          # Soil scoring (SoilGrids data)
│   │   ├── topography.py    # Slope/elevation/aspect scoring
│   │   ├── landcover.py     # Land cover compatibility scoring
│   │   ├── water.py         # Water proximity scoring
│   │   └── socioeconomic.py # Socioeconomic scoring
│   ├── suitability/
│   │   ├── overlay.py       # Weighted overlay aggregation engine
│   │   ├── classifier.py    # FAO suitability classification
│   │   └── sensitivity.py   # Sensitivity analysis on AHP weights
│   ├── data/
│   │   ├── downloaders.py   # Automated dataset downloaders
│   │   └── preprocessors.py # Clip, reproject, and align rasters
│   └── output/
│       ├── mapgen.py        # Map and chart generation (static + interactive)
│       └── report.py        # Text and HTML report generation
├── data/
│   ├── raw/                 # Downloaded raw datasets (git-ignored)
│   └── processed/           # Preprocessed rasters (git-ignored)
├── output/                  # Analysis results (git-ignored)
├── notebooks/               # Jupyter notebooks for exploration
├── pyproject.toml           # Project dependencies and metadata
└── .gitignore
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- [GDAL](https://gdal.org/) (required by `rasterio` and `fiona`)

### Setup

```bash
# Clone the repository
git clone https://github.com/cypher6783/GisLandUse.git
cd GisLandUse

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -e .

# For development (includes Jupyter)
pip install -e ".[dev]"
```

---

## 🚀 Usage

The project exposes a simple CLI with the following commands:

### 1. Download Data

Downloads all required datasets (WorldClim climate, SoilGrids, SRTM DEM, ESA land cover, etc.):

```bash
python -m src.main download
```

### 2. Preprocess Data

Clips rasters to Nigeria's boundary, reprojects to a common CRS, and aligns all layers:

```bash
python -m src.main preprocess
```

### 3. Run Suitability Analysis

Performs weighted overlay, classifies suitability, and generates all output maps and reports:

```bash
python -m src.main analyze
```

### 4. View AHP Weights

Displays the AHP pairwise matrix, derived weights, and consistency ratio:

```bash
python -m src.main ahp
```

### 5. Run Sensitivity Analysis

Evaluates how robust the suitability result is to changes in AHP weights:

```bash
python -m src.main sensitivity
```

---

## 📊 Outputs

After running `analyze`, the following files are saved to the `output/` directory:

| File | Description |
|---|---|
| `suitability_score.tif` | Continuous suitability score raster (0–1) |
| `suitability_class.tif` | Classified suitability raster (S1/S2/S3/N) |
| `suitability_map.png` | Static suitability class map |
| `score_map.png` | Continuous score visualization |
| `area_distribution.png` | Bar chart of area by suitability class |
| `weight_distribution.png` | AHP weight distribution chart |
| `suitability_interactive.html` | Interactive Folium map |
| `suitability_report.txt` | Plain-text analysis report |
| `suitability_report.html` | Full HTML report with embedded maps |
| `analysis_metadata.json` | Weights, AHP stats, and area statistics |

---

## 🧮 Methodology

### AHP (Analytic Hierarchy Process)

Criteria weights are derived from a pairwise comparison matrix using Saaty's AHP method. The **Consistency Ratio (CR)** is validated to ensure reliable judgements:

- **CR threshold**: ≤ 0.10 (10%) for a 6×6 matrix
- **Achieved CR**: ~0.016 ✅ (highly consistent)

### Weighted Overlay

Each criterion raster is scored on a normalized 0–1 scale, then combined as:

```
Suitability Score = Σ (weight_i × score_i)
```

Pixels with missing data are handled gracefully by normalizing weights for available criteria only.

### Sensitivity Analysis

The sensitivity module perturbs each criterion's weight by ±10–30% to assess how much the final classification changes, providing confidence bounds on the suitability map.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `rasterio` | Raster I/O and geospatial operations |
| `geopandas` | Vector data handling |
| `numpy` | Numerical array operations |
| `matplotlib` | Static map and chart generation |
| `folium` | Interactive HTML maps |
| `scipy` | Statistical computations |
| `pandas` | Tabular data processing |
| `xarray` | Multi-dimensional array support |
| `shapely` | Geometric operations |
| `pyproj` | Coordinate reference system management |

---

## 📄 License

This project is open-source. Feel free to use, modify, and distribute it with attribution.

---

## 🙏 Data Sources

- **Climate**: [WorldClim v2.1](https://www.worldclim.org/) — BIO variables at 30 arc-seconds
- **Soil**: [SoilGrids 250m](https://soilgrids.org/) — pH, SOC, clay, sand, bulk density
- **DEM/Topography**: [SRTM 90m](https://srtm.csi.cgiar.org/)
- **Land Cover**: [ESA WorldCover](https://esa-worldcover.org/)
- **Water Bodies**: [OpenStreetMap](https://www.openstreetmap.org/) via Geofabrik
- **Administrative Boundary**: [GADM](https://gadm.org/) — Nigeria Level 0
