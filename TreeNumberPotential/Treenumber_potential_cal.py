# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 17:17:39 2021

@author: LT
"""
import os
import numpy as np
import pandas as pd
from osgeo import gdal
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.utils import read_tiff, write_tiff

class DataProcessor:
    def __init__(self, in_label_dir, out_path, num_path):
        self.in_label_dir = in_label_dir
        self.out_path = out_path
        self.num_path = num_path
        self.out_other = os.path.join(out_path, 'Other')
        self.out_number = os.path.join(out_path, 'Number')
        os.makedirs(self.out_other, exist_ok=True)
        os.makedirs(self.out_number, exist_ok=True)
        plt.rcParams['font.sans-serif'] = 'Times New Roman'

    def process_data(self):
        namelist = [l for l in os.listdir(self.in_label_dir) if l.endswith('.tif')]
        for m in range(1):  # Example for processing only the first file
            filename = namelist[m]
            self.process_file(filename)

    def process_file(self, filename):
        # Implement the logic that was previously in the for loop
        proda_path = os.path.join("../result/TGS_score", "TGS_score.tif")
        tree_number_path = os.path.join("../dataset/TreeDensity", "TreeNumber_China.tif")
        label_path = os.path.join("../dataset/forest_label", "All_Trees.tif")

        score, geot, proj = read_tiff(proda_path)
        score[score < -10000] = -1

        Tree_number, geot, proj = read_tiff(tree_number_path)
        Tree_number[Tree_number < -99] = 0

        label,_,_ = read_tiff(label_path)
        label[label < -10000] = -1

        # Combine arrays and perform computations
        self.combine_and_compute(score, Tree_number, label, geot, proj, filename)

    def combine_and_compute(self, proda0, Tree_number, label, geot, proj, outname):
        proda1 = proda0[:, :, np.newaxis]
        Tree_number1 = Tree_number[:, :, np.newaxis]
        label = label[:, :, np.newaxis]
        
        set2 = np.concatenate((proda1, Tree_number1, label), axis=2)
        self.perform_calculations(set2, geot, proj, outname)

    def save_results(self, data, geot, proj, filename):
        output_path = os.path.join(self.out_number, filename)
        write_tiff(output_path, data, geot, proj)

    def perform_calculations(self, data, geot, proj, outname):
        proda = data[:, :, 0]  # Assuming proda values are in the first channel
        tree_number = data[:, :, 1]  # Tree number values
        label = data[:, :, 2]  # Label values

        # Prepare arrays to hold results
        result_median = np.zeros_like(tree_number, dtype=np.float32)
        result_75 = np.zeros_like(tree_number, dtype=np.float32)
        result_25 = np.zeros_like(tree_number, dtype=np.float32)
        
        # Assuming data length matches the required process length
        list2 = []
        min_samples = 500

        for pro in tqdm(range(10000)):  # Example range, adjust based on your data scale
            step = 10
            collected_samples = []
            
            while len(collected_samples) < min_samples and step < 10000:
                proda_min = pro * 10 - step
                proda_max = (pro + 1) * 10 + step
                relevant_indices = (proda >= proda_min) & (proda <= proda_max)
                collected_samples = tree_number[relevant_indices]
                if len(collected_samples) < min_samples:
                    step += 10  # Increase search range incrementally

            # Increase search range incrementally
            if len(collected_samples) > min_samples:
                lower_q = np.quantile(collected_samples, 0.25)
                upper_q = np.quantile(collected_samples, 0.75)
                median_q = np.median(collected_samples)

                # Store results in the output arrays
                result_median[relevant_indices] = median_q
                result_75[relevant_indices] = upper_q
                result_25[relevant_indices] = lower_q
                    
                    # Optionally collect results for reporting or logging
                list1 = [(proda_min + proda_max) / 2.0, lower_q, median_q, upper_q]
                list2.append(list1)

        # Save or log your results as needed
        self.save_results(result_median, geot, proj, outname + '_50.tif')
        self.save_results(result_75, geot, proj, outname + '_75.tif')
        self.save_results(result_25, geot, proj, outname + '_25.tif')
        
        # Optionally, save the summary statistics to an Excel file
        df = pd.DataFrame(list2, columns=['proda_range', '25%', '50%', '75%'])
        df.to_excel(os.path.join(self.out_other, outname + '_summary.xlsx'), index=False)

# Usage example:
if __name__ == '__main__':
    dp = DataProcessor(
        in_label_dir=r'../dataset/forest_type',
        out_path=r'../result/Tree_potential/potential_cal',
    )
    dp.process_data()
