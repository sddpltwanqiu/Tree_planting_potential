# -*- coding: utf-8 -*-
"""
Processing and preparing datasets.

@author: LT
"""
import os
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

class DataPreparation:
    def __init__(self, label_path, input_parameter_dir, output_dir):
        self.label_path = label_path
        self.input_parameter_dir = input_parameter_dir
        self.output_dir = output_dir
        plt.rcParams['font.sans-serif'] = 'Times New Roman'

    def read_tiff(self, path):
        """Utility function to read TIFF file."""
        dataset = gdal.Open(path)
        band = dataset.GetRasterBand(1)
        arr = band.ReadAsArray()
        geot = dataset.GetGeoTransform()
        proj = dataset.GetProjection()
        return arr, geot, proj

    def preprocess_data(self):
        """Process input data and prepare datasets."""
        label, geot, proj = self.read_tiff(self.label_path)
        label[label < 0] = -1
        h, w = label.shape

        paralist = [j for j in os.listdir(self.input_parameter_dir) if j.endswith('.tif')]
        num_p = len(paralist)
        para_mat = np.zeros((h, w, num_p), dtype=np.float32)

        for k, name in enumerate(paralist):
            pth = os.path.join(self.input_parameter_dir, name)
            img, _, _ = self.read_tiff(pth)
            img[img < -101] = np.NAN
            para_mat[:, :, k] = img

        all_data = np.concatenate((para_mat, label[:, :, None]), axis=2)
        all_data = all_data.reshape([-1, num_p + 1])
        all_data = all_data[~np.isnan(all_data).any(axis=1)]

        return all_data

    def save_splits(self, data, num_splits=5):
        """Split data into training and testing datasets and save them."""
        np.random.shuffle(data)
        splits = np.array_split(data, num_splits)

        for i in range(num_splits):
            test_set = splits[i]
            train_sets = np.concatenate(splits[:i] + splits[i+1:])
            
            fold_dir = os.path.join(self.output_dir, f'Fold_{i}')
            os.makedirs(fold_dir, exist_ok=True)
            
            np.save(os.path.join(fold_dir, f'train_{i}.npy'), train_sets)
            np.save(os.path.join(fold_dir, f'test_{i}.npy'), test_set)

if __name__ == '__main__':
    # Example usage
    dp = DataPreparation(
        label_path='../dataset/forest_label/Hansen_1km_th30.tif',
        input_parameter_dir='../dataset/environmental',
        output_dir='/home/liutang/Tree_Carbon/dataset_20240402/'
    )
    data = dp.preprocess_data()
    dp.save_splits(data)
