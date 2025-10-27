#  Air Quality Prediction — PM2.5 from Satellite AOD (Africa)

Estimate ground-level **PM2.5** concentrations from satellite **Aerosol Optical Depth (AOD)** and related atmospheric variables using **machine learning**.  
This project focuses on eight African cities — **Lagos (Nigeria)**, **Accra (Ghana)**, **Nairobi** & **Kisumu (Kenya)**, **Yaoundé (Cameroon)**, **Bujumbura (Burundi)**, **Kampala** & **Gulu (Uganda)**.

---

##  Project Background

Air pollution is the **world’s largest environmental health risk**, responsible for nearly **7 million premature deaths** annually.  
In sub-Saharan Africa, sparse monitoring networks make it difficult to track pollution levels and design effective policies.

This project uses **satellite-derived Aerosol Optical Depth (AOD)** and **machine-learning models** to estimate PM2.5 concentrations, filling gaps in ground-based data.  
The outputs can support:

- 🏙️ **Urban vulnerability profiling**
- 💚 **Public-health interventions**
- ⚖️ **Environmental justice**
- 🌦️ **Climate-change mitigation**
- 🤝 **Community empowerment**

Models will ultimately integrate into **AirQo’s digital platform**, giving citizens and policymakers access to reliable, localized air-quality information.

---

##  Objectives

1. **Model** PM2.5 using AOD and auxiliary predictors (meteorology, gases, land use, time, etc.).  
2. **Validate** predictions against ground-monitoring stations.   
3. **Document** data processing, modeling choices, and performance metrics.

---

##  Repository Structure
```
air_quality_prediction/
├─ src/ # source code for data prep, training, evaluation
├─ configs/ # configuration files (model params, paths, CV setup)
├─ data/ # local folder (raw/processed not committed to Git)
├─ outputs/ # trained models, evaluation results, submissions
├─ catboost_info/ # CatBoost training logs (auto-generated)
├─ scratch2.ipynb # exploration / prototyping notebook
├─ create_submission.py # script to generate predictions/submissions
├─ Makefile # convenience automation commands
└─ README.md
```


---

##  Dataset Summary

The initial dataset contains **80 columns** describing satellite and atmospheric variables across multiple sensors.

| Category | Examples | Description |
|-----------|-----------|-------------|
| **Identifiers** | `id`, `site_id` | Unique record and monitoring site IDs |
| **Geolocation** | `site_latitude`, `site_longitude`, `city`, `country` | Station/city coordinates and metadata |
| **Temporal** | `date`, `hour`, `month`, `day_of_week`, `year` | Time stamps for each observation |
| **Sulphur Dioxide (SO₂)** | `sulphurdioxide_so2_column_number_density`, `sulphurdioxide_cloud_fraction`, … | Vertical and slant column densities, AMF, cloud fraction, geometry |
| **Carbon Monoxide (CO)** | `carbonmonoxide_co_column_number_density`, `carbonmonoxide_h2o_column_number_density`, … | Trace gas columns and viewing angles |
| **Nitrogen Dioxide (NO₂)** | `nitrogendioxide_no2_column_number_density`, `nitrogendioxide_tropospheric_no2_column_number_density`, … | Tropospheric and stratospheric NO₂ measures |
| **Formaldehyde (HCHO)** | `formaldehyde_tropospheric_hcho_column_number_density`, `formaldehyde_cloud_fraction`, … | HCHO trace gas columns |
| **Ozone (O₃)** | `ozone_o3_column_number_density`, `ozone_o3_effective_temperature`, … | Ozone column data and temperature profiles |
| **Aerosols (UVAI / Height)** | `uvaerosolindex_absorbing_aerosol_index`, `uvaerosollayerheight_aerosol_optical_depth`, … | Aerosol index and layer height info |
| **Cloud Properties** | `cloud_cloud_fraction`, `cloud_cloud_top_height`, … | Cloud optical depth, height, albedo, geometry |
| **Ancillary** | `elevation` | Station altitude above sea level |

**Total columns:** ~80  
**Primary target variable:** `pm25` (to be derived or joined from ground monitors)  
**Data type mix:** numeric (float), categorical (city, country), temporal (date/time).

---

##  Modeling Approach

- **Model family:** Gradient boosting ( CatBoost, XGBoost, LightGBM).  
- **Inputs:** AOD, trace-gas columns, meteorological variables, geometry (zenith/azimuth angles).  
- **Targets:** Ground-truth PM2.5 concentrations.  
- **Validation:** K-fold, city-wise, and leave-one-station-out cross-validation.  
- **Metrics:** RMSE, MAE, R² per city/time segment.  
- **Outputs:** Predicted PM2.5 maps and CSV submissions.

---

## ⚙️ Installation & Setup

```
git clone https://github.com/JosephKiragu/air_quality_prediction.git
cd air_quality_prediction

# optional virtual environment
python -m venv .venv && source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

## Train model
```
python -m src.train \
  --config configs/catboost.yaml \
  --out outputs/models/catboost.pkl
```


## Evaluate
```
python -m src.evaluate \
  --model outputs/models/catboost.pkl \
  --test data/processed/test.parquet \
  --metrics_out outputs/metrics/catboost.json
```

## Generate submission
```
python create_submission.py \
  --model outputs/models/catboost.pkl \
  --input data/processed/predict.parquet \
  --out outputs/submissions/predictions.csv
```

## ⚖️ Evaluation Criteria

Accuracy: RMSE, MAE, and R² on held-out sets.

Generalization: Per-city and per-station breakdowns.

Robustness: Temporal and spatial extrapolation checks.

Interpretability: Feature-importance plots and SHAP summaries.

##  Configuration

All experiment parameters (paths, hyperparameters, folds) are stored in configs/ (YAML/JSON).
Each run saves its outputs, metrics, and configuration snapshot under outputs/.

##  Reproducibility

Deterministic random seeds

Versioned data and configs

Logged parameters and metrics

Clear model registry under outputs/models/
