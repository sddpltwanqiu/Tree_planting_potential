# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 22:37:08 2022
@author: DELL
"""

from osgeo import gdal
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pickle
from ..utils.utils import read_tiff, write_tiff
from warnings import simplefilter

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)
plt.rcParams['font.sans-serif'] = 'Times New Roman'


work_path = '../Carbon_layer/Result_spawn/'
type_list = ['EB','DB','DN','EN']

for t in range(4):
    Type = type_list[t]

    proda1,_,_ = read_tiff('../result/TGS_score/TGS_score.tif')
    proda1[proda1 < -10000] = -1

    in_number_path = '../result/Tree_potential/potential_rm_crop/Max_number'+ Type

    MASK = '../result/Tree_potential/potential_rm_crop/MASK/SHOULD_MASK'+ Type
    New_MASK =  '../result/Tree_potential/potential_rm_crop/MASK/NEW_MASK'+ Type
    Old_MASK = '../result/Tree_potential/potential_rm_crop/MASK/OLD_MASK'+ Type

    model = '../dataset/Carbon_layer/Result_spawn/Tree_number_carbon_box/'+Type

    above_mean =  '../dataset/Carbon_layer/Result_spawn/Max_Carbon/Mean/above'
    below_mean = '../dataset/Carbon_layer/Result_spawn/Max_Carbon/Mean/below'

    above_max = '../dataset/Carbon_layer/Result_spawn/Max_Carbon/Max/above'
    below_max ='../dataset/Carbon_layer/Result_spawn/Max_Carbon/Max/below'

    above_min = '../dataset/Carbon_layer/Result_spawn/Max_Carbon/Min/above'
    below_min = '../dataset/Carbon_layer/Result_spawn/Max_Carbon/Min/below'
    
    os.makedirs(above_mean+ '\\SHOULD', exist_ok=True)
    os.makedirs(below_mean+ '\\SHOULD', exist_ok=True)
    os.makedirs(above_max+ '\\SHOULD', exist_ok=True)
    os.makedirs(below_max+ '\\SHOULD', exist_ok=True)
    os.makedirs(above_min+ '\\SHOULD', exist_ok=True)
    os.makedirs(below_min+ '\\SHOULD', exist_ok=True)
    
    os.makedirs(above_mean + '\\NEW', exist_ok=True)
    os.makedirs(below_mean + '\\NEW', exist_ok=True)
    os.makedirs(above_max + '\\NEW', exist_ok=True)
    os.makedirs(below_max + '\\NEW', exist_ok=True)
    os.makedirs(above_min + '\\NEW', exist_ok=True)
    os.makedirs(below_min + '\\NEW', exist_ok=True)
    
    os.makedirs(above_mean + '\\OLD', exist_ok=True)
    os.makedirs(below_mean + '\\OLD', exist_ok=True)
    os.makedirs(above_max + '\\OLD', exist_ok=True)
    os.makedirs(below_max + '\\OLD', exist_ok=True)
    os.makedirs(above_min + '\\OLD', exist_ok=True)
    os.makedirs(below_min + '\\OLD', exist_ok=True)

    file_list = [l for l in os.listdir(in_number_path) if l[-4:] == '.tif']

    for i in range(len(file_list)):
        file_name = os.path.join(in_number_path, file_list[i])

        number, geot, proj = read_tiff(file_name)
        number[number < -99] = 0

        mask_name = os.path.join(MASK, file_list[i])
        mask_name_new = os.path.join(New_MASK, file_list[i])
        mask_name_old = os.path.join(Old_MASK, file_list[i])
    
        mask_data, _, _ = read_tiff(mask_name)
        mask_data[mask_data < -99] = -1
        mask_data_new, _, _ = read_tiff(mask_name_new)
        mask_data_new[mask_data_new < -99] = -1
        mask_data_old, _, _ = read_tiff(mask_name_old)
        mask_data_old[mask_data_old < -99] = -1

        human_path = '../dataset/LandCover/Humanbeing_FQ.tif'
        water_path = '../dataset/LandCover/Water_mask.tif'
        water_mask, _, _ = read_tiff(water_path)
        water_mask[water_mask < -99] = -1

        human_being, _, _ = read_tiff(human_path)
        human_being[human_being < -99] = -1

        above_stuo,_,_ = read_tiff(os.path.join(work_path, 'AGBC_2010.tif'))
        above_stuo[above_stuo < -99] = 0
        
        below_stuo,_,_ = read_tiff(os.path.join(work_path, 'BGBC_2010.tif'))
        below_stuo[below_stuo < -99] = 0

        result_above = np.zeros_like(number)
        result_below = np.zeros_like(number)
        result_above_min = np.zeros_like(number)
        result_below_min = np.zeros_like(number)
        result_above_max = np.zeros_like(number)
        result_below_max = np.zeros_like(number)
    
        for pro in tqdm(range(40)):
            proda_min = pro * 250
            proda_max = (pro+1) * 250
            number_1 = ((proda1>=proda_min)&(proda1<=proda_max)).astype(np.uint8)
        
            with open(model + '\\model_mean\\above_' + str(pro) +'.pkl', 'rb') as f:
                afunc = pickle.load(f)
            with open(model + '\\model_mean\\below_' + str(pro) +'.pkl', 'rb') as f:
                bfunc = pickle.load(f)
            
            result_above_1 = number_1*(afunc(number))
            result_below_1 = number_1*(bfunc(number))
            result_above = result_above + result_above_1
            result_below = result_below + result_below_1
        
            with open(model + '\\model_max\\above_' + str(pro) +'.pkl', 'rb') as f:
                afunc_max = pickle.load(f)
            with open(model + '\\model_max\\below_' + str(pro) +'.pkl', 'rb') as f:
                bfunc_max = pickle.load(f)  
            
            result_above_1 = number_1*(afunc_max(number))
            result_below_1 = number_1*(bfunc_max(number))
            result_above_max = result_above_max + result_above_1
            result_below_max = result_below_max + result_below_1
            
            with open(model + '\\model_min\\above_' + str(pro) +'.pkl', 'rb') as f:
                afunc_min = pickle.load(f)
            with open(model + '\\model_min\\below_' + str(pro) +'.pkl', 'rb') as f:
                bfunc_min = pickle.load(f)  
        
            result_above_1 = number_1*(afunc_min(number))
            result_below_1 = number_1*(bfunc_min(number))
            result_above_min = result_above_min + result_above_1
            result_below_min = result_below_min + result_below_1
    
        result_above = result_above*(1-water_mask)*human_being
        result_below = result_below*(1-water_mask)*human_being
        result_above_min = result_above_min*(1-water_mask)*human_being
        result_below_min = result_below_min*(1-water_mask)*human_being
        result_above_max = result_above_max*(1-water_mask)*human_being
        result_below_max = result_below_max*(1-water_mask)*human_being
    
        result_above_out = (result_above - above_stuo)*mask_data
        result_below_out = (result_below - below_stuo)*mask_data
        ofile_a = os.path.join(above_mean + '\\SHOULD',  file_list[i])
        ofile_b = os.path.join(below_mean + '\\SHOULD',  file_list[i])
        write_tiff(ofile_a,result_above_out,geot,proj)
        write_tiff(ofile_b,result_below_out,geot,proj)
    
        result_above_min_out = (result_above_min - above_stuo) * mask_data
        result_below_min_out = (result_below_min - below_stuo) * mask_data    
        ofile_a = os.path.join(above_min + '\\SHOULD',  file_list[i])
        ofile_b = os.path.join(below_min + '\\SHOULD',  file_list[i])
        write_tiff(ofile_a,result_above_min_out,geot,proj)
        write_tiff(ofile_b,result_below_min_out,geot,proj)
    
        result_above_max_out = (result_above_max - above_stuo)*mask_data
        result_below_max_out = (result_below_max - below_stuo)*mask_data
        ofile_a = os.path.join(above_max + '\\SHOULD',  file_list[i])
        ofile_b = os.path.join(below_max + '\\SHOULD',  file_list[i])
        write_tiff(ofile_a,result_above_max_out,geot,proj)
        write_tiff(ofile_b,result_below_max_out,geot,proj)
    
        #new
        result_above_new = (result_above - above_stuo)*mask_data_new
        result_below_new = (result_below - below_stuo)*mask_data_new
        ofile_a = os.path.join(above_mean + '\\NEW',  file_list[i])
        ofile_b = os.path.join(below_mean + '\\NEW',  file_list[i])
        write_tiff(ofile_a,result_above_new,geot,proj)
        write_tiff(ofile_b,result_below_new,geot,proj)
    
        result_above_min_new =  (result_above_min - above_stuo)*mask_data_new
        result_below_min_new = (result_below_max - below_stuo)*mask_data_new    
        ofile_a = os.path.join(above_min + '\\NEW',  file_list[i])
        ofile_b = os.path.join(below_min + '\\NEW',  file_list[i])
        write_tiff(ofile_a,result_above_min_new,geot,proj)
        write_tiff(ofile_b,result_below_min_new,geot,proj)
        
        result_above_max_new = (result_above_max - above_stuo)*mask_data_new
        result_below_max_new = (result_below_max - below_stuo)*mask_data_new
        ofile_a = os.path.join(above_max + '\\NEW',  file_list[i])
        ofile_b = os.path.join(below_max + '\\NEW',  file_list[i])
        write_tiff(ofile_a,result_above_max_new,geot,proj)
        write_tiff(ofile_b,result_below_max_new,geot,proj)
        
        #old
        result_above_old = (result_above - above_stuo)*mask_data_old
        result_below_old = (result_below - below_stuo)*mask_data_old
        ofile_a = os.path.join(above_mean + '\\OLD',  file_list[i])
        ofile_b = os.path.join(below_mean + '\\OLD',  file_list[i])
        write_tiff(ofile_a,result_above_old,geot,proj)
        write_tiff(ofile_b,result_below_old,geot,proj)
        
        result_above_min_old =  (result_above_min - above_stuo)*mask_data_old
        result_below_min_old =  (result_below_max - below_stuo)*mask_data_old    
        ofile_a = os.path.join(above_min + '\\OLD',  file_list[i])
        ofile_b = os.path.join(below_min + '\\OLD',  file_list[i])
        write_tiff(ofile_a,result_above_min_old,geot,proj)
        write_tiff(ofile_b,result_below_min_old,geot,proj)
        
        result_above_max_old = (result_above_max - above_stuo)*mask_data_old
        result_below_max_old = (result_below_max - below_stuo)*mask_data_old
        ofile_a = os.path.join(above_max + '\\OLD',  file_list[i])
        ofile_b = os.path.join(below_max + '\\OLD',  file_list[i])
        write_tiff(ofile_a,result_above_max_old,geot,proj)
        write_tiff(ofile_b,result_below_max_old,geot,proj)
    

    
    
    
    
    


