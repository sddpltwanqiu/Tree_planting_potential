# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 09:25:31 2021

@author: LT
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from tqdm import tqdm
from ..utils.utils import read_tiff, write_tiff
# Paths
TreeNumber_path = '../dataset/TreeDensity/TreeNumber.tif'
ForestType_path = '../dataset/forest_type'
Humanbeing_path = '../dataset/LandCover/Humanbeing_FQ.tif'
Water_path = '../dataset/LandCover/Water_mask.tif'
input_path = '../result/Tree_potential/potential_cal'
result_path = '../result/Tree_potential/potential_rm_crop'

# Result subdirectories
subpaths = ['Should_add', 'Max_number', 'New_add', 'Old_add']
oupaths = [os.path.join(result_path, sub) for sub in subpaths]
for oupath in oupaths:
    if not os.path.exists(oupath):
        os.makedirs(oupath)

# Data loading
Tree_number, geot, proj = read_tiff(TreeNumber_path)
Tree_number[Tree_number < -99] = 0

humanbeing,_,_ = read_tiff(Humanbeing_path)
humanbeing[humanbeing < -10000] = -1

water,_,_ = read_tiff(Water_path)
water[water < -10000] = -1
water = 1 - water  # Inverting the water mask

# Process each file in the input directory
file_list = [f for f in os.listdir(input_path) if f.endswith('.tif')]
for name in tqdm(file_list):
    ipath = os.path.join(input_path, name)

    All_tree_label,_,_ = read_tiff(ipath)
    All_tree_label[All_tree_label < -10000] = -1

    ft_name = name[:-7] + '.tif'
    ft_path = os.path.join(ForestType_path, ft_name)

    ft_label,_,_ = read_tiff(ft_path)
    ft_label[ft_label < -10000] = -1
    
    w, h = Tree_number.shape
    result = np.zeros((w, h, 2))
    result[:, :, 0] = All_tree_label
    result[:, :, 1] = Tree_number
    
    tree_number = np.max(result, axis=2)
    should_add = tree_number - Tree_number
    new_add = np.where(should_add == tree_number, should_add, 0)
    old_add = should_add - new_add
    
    # File paths for output
    base_name = name[:-4]
    paths = [os.path.join(oup, base_name + suffix) for oup, suffix in zip(oupaths, ['_max_number.tif', '_should_add.tif', '_afforest.tif', '_densify.tif'])]
    
    # Applying masks
    masks = humanbeing * water * ft_label
    outputs = [x * masks for x in [tree_number, should_add, old_add, new_add]]
    
    # Writing outputs
    for path, output in zip(paths, outputs):
        write_tiff(path, output, geot, proj)
