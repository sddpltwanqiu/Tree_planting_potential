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
The project is divided into three stages: Tree Growth Suitability (TGS) Score calculation, tree planting potential assessment, and carbon storage estimation. Due to the need for some operations to be completed in ARCGIS 10.8, each calculation stage is relatively independent, and users can download the results from each stage as needed.
Navigate to a specific model directory (e.g., CarbonPotential or TreeNumberPotential) and execute the corresponding Python scripts to generate analysis results.

```bash
cd CarbonPotential
cd TreeNumberPotential
cd 
```

## Data
Download the necessary data files from the provided links or specified sources (See Data Availability in the paper).
Please ensure that the necessary datasets are placed in the appropriate directories before running the analysis.

## Results
The result can be found at https://doi.org/10.6084/m9.figshare.25707414.

## Contributing
Contributions to this project are welcome! If you have any suggestions, bug fixes, or new features, please submit a pull request or open an issue.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

