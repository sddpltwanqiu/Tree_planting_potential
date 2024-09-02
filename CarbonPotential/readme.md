#  Carbon Fluctuation Modeling

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Usage](#usage)
- [Results](#results)

## Introduction

This branch is focused on establishing the relationship between tree density and biomass carbon storage to model carbon fluctuations caused by changes in tree density.

## Data

- **Biomass carbon maps:** The aboveground and belowground biomass carbon maps (Spwan et al., 2019) are from https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1763, https://doi.org/10.6084/m9.figshare.21931161.v1 (Chen et al., 2023) and https://doi.org/10.1098/rstb.2019.0128 (Soto-Navarro et al.. 2020).
- **TGS score data:** This data can be obtained through the calculations in the TGS_score_model step or downloaded from https://doi.org/10.6084/m9.figshare.25707414.
- **Forest types classification dataset:** Forest types classification dataset (MCD12Q1: Type5 Plant Functions Types) can be obtained at https://lpdaac.usgs.gov/products/mcd12q1v006/.
- **Tree density:** The global tree density map is available at http://elischolar.library.yale.edu/yale_fes_data/1/.

## Usage

Please ensure that the necessary datasets are placed in the appropriate directories before running the analysis.
It includes three phase:

First, extract tree density and biomass carbon storage data based on the TGS intervals， including maximum, minimum and medium value.
```bash
cd CarbonPotential
python Step1_Carbon_statistic.py
```

Second, establish the relationship models between tree density and biomass carbon storage.
```bash
python Step2_Carbon_interpolate_fit.py
```

Third, Use the model established in the previous step for global inference to estimate carbon storage.
```bash
python Step3_Carbon_model_inference.py
```

## Results
The result can be found at https://doi.org/10.6084/m9.figshare.25707414.
