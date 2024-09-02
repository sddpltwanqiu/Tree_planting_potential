# Carbon Sequestration Potential of Tree Planting in China
This repository contains the code for the project "Carbon Sequestration Potential of Tree Planting in China."

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project is focused on identifying and evaluating areas suitable for tree planting. By leveraging remote sensing data and advanced geospatial analysis techniques, the project aims to provide valuable insights into regions where afforestation efforts can be most effective. 

## Features

- **Environmental Variable Analysis:** Evaluates the suitability of tree planting using multiple environmental variables and cross-validation methods.
- **Tree Density Potential Assessment:** Assesses the potential for increased tree density based on the combination of TGS scores and existing tree density.
- **Carbon Fluctuation Modeling:** Establishes the relationship between tree density and biomass carbon storage to model carbon fluctuations caused by changes in tree density.

## Installation

To use the code in this repository, clone the repository and install the required dependencies. Pleaes ensure your environment meets the following requirements:
Python version 3.7.3 or higher
PyTorch version 1.0.0 or higher

```bash
git clone https://github.com/sddpltwanqiu/Tree_planting_potential.git
```

## Usage
The project is divided into three stages: Tree Growth Suitability (TGS) Score calculation, tree planting potential assessment, and carbon storage estimation. 
Navigate to a specific model directory (e.g., CarbonPotential or TreeNumberPotential) and execute the corresponding Python scripts to generate analysis results.
Due to the need for some operations to be completed in ARCGIS 10.8, each calculation stage is relatively independent, and users can download the results from each stage as needed.

For TGS Score calculation, need Environmental Variable data with 1km resolution (See Data Availability in the manuscript), land cover (http://data.ess.tsinghua.edu.cn/fromglc2017v1.html), forest distribution (https://glad.earthengine.app/view/global-forest-change) and validation data (https://doi.org/10.11922/sciencedb.j00076.00091)

```bash
cd TGS_score_model
```
For tree planting potential assessment, need TGS score, Forest types classification dataset (MCD12Q1: Type5 Plant Functions Types) and tree density data (Nature, 2015):
```bash
cd TreeNumberPotential
```
For carbon storage estimation, need the aboveground and belowground biomass carbon maps (https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1763)
```bash
cd CarbonPotential
```

## Data
Download the necessary source data files from the provided links or specified sources (See Data Availability in the manuscript).
Please ensure that the necessary datasets are placed in the appropriate directories before running the analysis.

## Results
The result can be found at https://doi.org/10.6084/m9.figshare.25707414.

## Contributing
Contributions to this project are welcome! If you have any suggestions, bug fixes, or new features, please submit a pull request or open an issue.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

