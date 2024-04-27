# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 17:17:39 2021

@author: LT
"""
from osgeo import gdal
import numpy as np
from tqdm import tqdm
from ..utils.utils import read_tiff, write_tiff
#import torch
#from torch import nn
#from scipy.stats import kde
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from numpy import log

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

plt.rcParams['font.sans-serif'] = 'Times New Roman'

work_path = '../Carbon_layer/Result_spawn'

in_label = '../dataset/forest_type'
cal_label = '../dataset/forest_type'
in_proda = '../result/TGS_score'

max_number = '../result/TGS_score'

out_other = work_path + '/Tree_number_carbon_sta'

if os.path.exists(out_other) == False:
    os.makedirs(out_other)

namelist = [l for l in os.listdir(in_label) if l[-4:] == '.tif']

for m in range(len(namelist)):
    outname = namelist[m]

    Tree_number, geot, proj = read_tiff('../dataset/TreeDensity/TreeNumber_China.tif')
    Tree_number[Tree_number < -99] = 0
    Tree_number1 = Tree_number[:,:,np.newaxis]

    pth = os.path.join(in_label,outname)
    label,_,_ = read_tiff(pth)
    label[label < -10000] = -1
    label = label[:,:,np.newaxis]

    proda0,_,_ = read_tiff('../result/TGS_score/TGS_score.tif')
    proda0[proda0 < -10000] = -1
    proda1 = proda0[:,:,np.newaxis]
    
    above_carbon0,_,_ = read_tiff( os.path.join(work_path, 'AGBC_2010.tif'))
    above_carbon0[above_carbon0 < -99] = 0
    above_carbon1 = above_carbon0[:,:,np.newaxis]

    below_carbon0,_,_ = read_tiff( os.path.join(work_path, 'BGBC_2010.tif'))
    below_carbon0[below_carbon0 < -99] = 0
    below_carbon1 = below_carbon0[:,:,np.newaxis]

    set2 = np.concatenate((above_carbon1, below_carbon1,Tree_number1,proda1, label),axis=2)
    h,w,b = set2.shape 
    set1 = set2.reshape(-1,b)
    set1 = np.array(set1[list(np.where(set1[:,-1]==1))])
    set1 = set1[0,:,:]
    
    list2=[]
 
    for pro in tqdm(range(40)):
        pro =pro
        
        proda_min = pro * 250
        proda_max = (pro+1) * 250
        #proda_ok = set1[np.where((set1[:,-2]>=proda_min)&(set1[:,-2]<=proda_max))]
        if len(set1[:,-2][np.where((set1[:,-2]>=proda_min)&(set1[:,-2]<=proda_max))])<20000:
            length = 20000
        else:
            length = len(set1[:,-2][np.where((set1[:,-2]>=proda_min)&(set1[:,-2]<=proda_max))])
        
        q = np.argsort(np.abs(set1[:,-2] - ((proda_max+proda_min)/2.0)))
        
        number_train = set1[:,-3][q[:length]]
        max_num = number_train.max()
        a_carbon_train = set1[:,0][q[:length]]
        b_carbon_train = set1[:,1][q[:length]]
        
        train_data = np.vstack((a_carbon_train, b_carbon_train, number_train)).T
        train_data = (train_data[~np.isnan(train_data).any(axis=1),:])
        
        list_pro = []
        th = 500
        for tree_num in range(int(max_num/10)):
            number_min = tree_num * 10
            number_max = (tree_num+1) * 10

            q = np.argsort(np.abs(train_data[:,2] - ((number_max+number_min)/2.0)))
            
            a_carbon_train = train_data[:,0][np.where((train_data[:,2]>=number_min)&(train_data[:,2]<=number_max))]
            b_carbon_train = train_data[:,1][np.where((train_data[:,2]>=number_min)&(train_data[:,2]<=number_max))]

            if len(a_carbon_train)<th:
                a_carbon_train = train_data[:,0][q[:th]]
                b_carbon_train = train_data[:,1][q[:th]]
            
            if len(a_carbon_train)==0:
                continue

            a_min_value = np.quantile(a_carbon_train,0.05,interpolation='higher')
            a_mean_value = np.nanmean(a_carbon_train)
            a_median_value = np.nanmedian(a_carbon_train)
            a_max_value = np.quantile(a_carbon_train,0.95,interpolation='higher')
            a_std = np.std(a_carbon_train)
            
            b_min_value = np.quantile(b_carbon_train,0.05,interpolation='higher')
            b_mean_value = np.nanmean(b_carbon_train)
            b_median_value = np.nanmedian(b_carbon_train)
            b_max_value = np.quantile(b_carbon_train,0.95,interpolation='higher')
            b_std = np.std(b_carbon_train)
            
            list_ = [(number_min+number_max)/2, a_min_value,b_min_value,a_mean_value, b_mean_value,a_median_value,b_median_value,a_max_value,b_max_value, a_std, b_std]
            list_pro.append(list_)
        
     
        list_pro = list_pro
        name1 = namelist[m][:-4]+'_'+str(pro)+'_carbon.xlsx'
        columns=['tree','a_min','b_min','a_mean','b_mean','a_median','b_median','a_max','b_max','a_std', 'b_std']
        name1 = os.path.join(out_other,name1)
        df = pd.DataFrame(list_pro,columns=columns)#dict(zip(columns,list_pro)) )
        df.to_excel(name1, index=False)