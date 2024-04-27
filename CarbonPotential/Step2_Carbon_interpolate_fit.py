# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 12:50:53 2022

@author: DELL
"""

import os 
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import numpy as np
from numpy.lib.scimath import log
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp1d

work_path = '../Carbon_layer/Result_spawn/'

#forest_type = ['DB','DN','EB','EN']
forest_path = 'DB'
inpth = work_path +'/Tree_number_carbon_sta/' + forest_path
oupth = work_path +'/Tree_number_carbon_box/' + forest_path
file_list = os.listdir(inpth)

boxes = 20
above = np.zeros((boxes,len(file_list)+1))
below = np.zeros((boxes,len(file_list)+1))

above_min = np.zeros((boxes,len(file_list)+1))
below_min = np.zeros((boxes,len(file_list)+1))
above_max = np.zeros((boxes,len(file_list)+1))
below_max = np.zeros((boxes,len(file_list)+1))

for i in tqdm(range(len(file_list))):
    name = file_list[i]
    pth1 = os.path.join(inpth,name)
    data = pd.read_excel(pth1)
    data = data.to_numpy()
    size = 250
    data = data[0:7000,:]
    
    for j in range(boxes):
        patch = data[j*size:(j+1)*size,:]
        tree = patch[:,0]
        tree_box = np.median(tree)
        tree_max = np.max(tree)
        
        if j == boxes-1:
            tree_box = tree_max
        
        above[j,0] = tree_box
        below[j,0] = tree_box
        above_min[j,0] = tree_box
        below_min[j,0] = tree_box
        above_max[j,0] = tree_box
        below_max[j,0] = tree_box
        
        
        above[j,i+1] = np.mean(data[j*size:(j+1)*size,3])
        below[j,i+1] = np.mean(data[j*size:(j+1)*size,4])
        above_min[j,i+1] = np.min(data[j*size:(j+1)*size,3])
        below_min[j,i+1] = np.min(data[j*size:(j+1)*size,4])
        above_max[j,i+1] = np.max(data[j*size:(j+1)*size,3])
        below_max[j,i+1] = np.max(data[j*size:(j+1)*size,4])
        
    # above mean
    ax = above[:,0]
    ay = above[:,i+1]
    afunc = interp1d(ax, ay, kind='linear')
    ooname = oupth+'/model_mean/above_'+str(i)+'.pkl'
    with open(ooname, 'wb') as f:
        pickle.dump(afunc, f, pickle.HIGHEST_PROTOCOL)
        
    # above min
    ax_min = above_min[:,0]
    ay_min = above_min[:,i+1]
    afunc_min = interp1d(ax_min, ay_min, kind='linear')
    ooname = oupth+'/model_min/above_'+str(i)+'.pkl'
    with open(ooname, 'wb') as f:
        pickle.dump(afunc_min, f, pickle.HIGHEST_PROTOCOL)
    
    # above max
    ax_max = above_max[:,0]
    ay_max = above_max[:,i+1]
    afunc_max = interp1d(ax_max, ay_max, kind='linear')
    ooname = oupth+'/model_max/above_'+str(i)+'.pkl'
    with open(ooname, 'wb') as f:
        pickle.dump(afunc_max, f, pickle.HIGHEST_PROTOCOL)
        
    #plot
    xHat = np.linspace(min(ax), max(ax), num=60000)
    yHat = afunc(xHat)
    yHat_min = afunc_min(xHat)
    yHat_max = afunc_max(xHat)
    
    plt.figure()
    plt.plot(ax, ay, 'o')
    plt.plot(xHat, yHat, '-')
    plt.plot(xHat, yHat_min, '-')
    plt.plot(xHat, yHat_max, '-')
    
    figname =  oupth+'/figure/above_'+str(i)+'.jpg'
    plt.tick_params(labelsize=12)    
    plt.xlabel("Number of Trees",fontsize=12)
    plt.ylabel("Aboveground Biomass Carbon(MgC)",fontsize=12)
    plt.savefig(figname,
            dpi=300,bbox_inches = 'tight')
    
    # below mean
    bx = below[:,0]
    by = below[:,i+1]

    bfunc = interp1d(bx, by, kind='linear')
    ooname = oupth+'/model_mean/below_'+str(i)+'.pkl'
    with open(ooname, 'wb') as f:
        pickle.dump(bfunc, f, pickle.HIGHEST_PROTOCOL)
        
    # below min
    bx_min = below_min[:,0]
    by_min = below_min[:,i+1]
    bfunc_min = interp1d(bx_min, by_min, kind='linear')
    ooname = oupth+'/model_min/below_'+str(i)+'.pkl'
    with open(ooname, 'wb') as f:
        pickle.dump(bfunc_min, f, pickle.HIGHEST_PROTOCOL)
    
    # above max
    bx_max = below_max[:,0]
    by_max = below_max[:,i+1]
    bfunc_max = interp1d(bx_max, by_max, kind='linear')
    ooname = oupth+'/model_max/below_'+str(i)+'.pkl'
    with open(ooname, 'wb') as f:
        pickle.dump(bfunc_max, f, pickle.HIGHEST_PROTOCOL)
        
    #plot
    xHat = np.linspace(min(bx), max(bx), num=60000)
    yHat = bfunc(xHat)
    yHat_min = bfunc_min(xHat)
    yHat_max = bfunc_max(xHat)
    
    plt.figure()
    plt.plot(bx, by, 'o')
    plt.plot(xHat, yHat, '-')
    plt.plot(xHat, yHat_min, '-')
    plt.plot(xHat, yHat_max, '-')
    
    figname =  oupth+'/figure/below_'+str(i)+'.jpg'
    plt.tick_params(labelsize=12)    
    plt.xlabel("Number of Trees",fontsize=12)
    plt.ylabel("Belowground Biomass Carbon(MgC)",fontsize=12)
    plt.savefig(figname,
            dpi=300,bbox_inches = 'tight')
    
    


        