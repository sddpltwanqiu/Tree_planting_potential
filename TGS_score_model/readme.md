#  Tree Growth Suitability (TGS) Score calculation

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Usage](#usage)
- [Results](#results)

## Introduction

This branch is focused on evaluating the suitability of tree planting using multiple environmental variables and cross-validation methods.

## Data

- **Environmental Variable data:** The nine soil variables can be found at https://soilgrids.org/; The three topographic variables can be found at https://research.utwente.nl/en/publications/hole-filled-srtm-for-the-globe-version-4-data-grid; The annual solar radiation data can be found at https://doi.org/10.6084/m9.figshare.c.4891302, The aridity index data can be found at https://doi.org/10.5281/zenodo.10074189. Annual maximum land surface temperature and annual minimum land surface temperature are available at https://lpdaac.usgs.gov/products/mod11a1v006/. Annual snow cover index can be found at https://nsidc.org/data/mod10a2/versions/5.
- **Land cover:** The land cover and land use change dataset is available at http://data.ess.tsinghua.edu.cn/fromglc2017v1.html (including urban and agricultural land data).
- **Forest distribution:** The Hansen Global Forest Change 2000-2022 Data is available at https://glad.earthengine.app/view/global-forest-change; 
- **Validation data :** The catalog of 816 forest parks can be found at www.gisrs.cn. All plantation forest field survey plots are available at https://doi.org/10.11922/sciencedb.j00076.00091.

## Usage

Please ensure that the necessary datasets are placed in the appropriate directories before running the analysis.

For data prepare, 
```bash
cd TGS_score_model
python TGS_dataprepare.py
```

For training TGS Score model, 
```bash
python TGS_trainer.py
```

For testing TGS Score model, 
```bash
python TGS_test.py
```

Using TGS Score model to map the whole China, 
```bash
python TGS_inference.py
```

## Results
The result can be found at https://doi.org/10.6084/m9.figshare.25707414.
