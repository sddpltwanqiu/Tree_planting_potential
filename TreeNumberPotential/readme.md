#  Tree Density Potential Assessment:

## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Usage](#usage)
- [Results](#results)

## Introduction

This branch is focused on assessing the potential for increased tree density based on the combination of TGS scores and existing tree density.

## Data

- **TGS score data:** This data can be obtained through the calculations in the TGS_score_model step or downloaded from https://doi.org/10.6084/m9.figshare.25707414.
- **Forest types classification dataset:** Forest types classification dataset (MCD12Q1: Type5 Plant Functions Types) can be obtained at https://lpdaac.usgs.gov/products/mcd12q1v006/.
- **Tree density:** The global tree density map is available at http://elischolar.library.yale.edu/yale_fes_data/1/.

## Usage

Please ensure that the necessary datasets are placed in the appropriate directories before running the analysis.

For data prepare, 
```bash
cd TreeNumberPotential
python Treenumber_potential_cal.py
### Compare with the current situation to calculate the actual potential, remove the potential in farmlands and urban areas, and c
python TreePotential_rm_cropcity.py
```

## Results
The result can be found at https://doi.org/10.6084/m9.figshare.25707414.
