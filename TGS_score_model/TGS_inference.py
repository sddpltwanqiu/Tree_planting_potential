# -*- coding: utf-8 -*-
"""
Processing and preparing datasets.

@author: LT
"""
import os
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import torch
from ..utils.utils import read_tiff, write_tiff
from utils import net
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils.loss_new import F1_Loss, Precision_loss
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'

class DataPreparation:
    def __init__(self, label_path, input_parameter_dir):
        self.label_path = label_path
        self.input_parameter_dir = input_parameter_dir
        plt.rcParams['font.sans-serif'] = 'Times New Roman'

    def preprocess_data(self):
        """Process input data and prepare datasets."""
        label, geot, proj = read_tiff(self.label_path)
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
        return self.normlize(para_mat), geot, proj
    
    def normlize(self, para_mat):
        mean = np.load('../dataset/environmental/mean_value.npy')
        std= np.load('../dataset/environmental/std_value.npy')
        para_mat= (para_mat - mean) / std
        return para_mat


def go(input_data,model_path,batch):
    if not torch.cuda.is_available():
        raise Exception("GPU Error!")

    checkpoint = torch.load(model_path)
    state = checkpoint['model_state_dict']
    model = net.MLPNet(18, 64, 32, 2)
    model = torch.nn.DataParallel(model).cuda().eval()
    model.load_state_dict(state)

    def parallelseg(npy):
        npy = npy.reshape((-1, 18))
        npy = torch.from_numpy(npy).type(torch.FloatTensor)
        batch_max = npy.size()[0]

        if (batch_max >= batch):
            softmax = []
            for idx in range(batch_max//batch):
                input = npy[idx*batch:(idx+1)*batch, :]
                with torch.no_grad():
                    # F.softmax(model(input.cuda()), dim=1)
                    # model(input.cuda())
                    output = F.softmax(model(input.cuda()), dim=1)
                softmax.append(output.cpu())
            input = npy[(idx+1)*batch:batch_max, :]
            
            with torch.no_grad():
                # F.softmax(model(input.cuda()), dim=1)
                # model(input.cuda())
                output = F.softmax(model(input.cuda()), dim=1)
            softmax.append(output.cpu())
            softmax = torch.cat(softmax, dim=0).numpy()
        else:
            with torch.no_grad():
                # F.softmax(model(npy.cuda()), dim=1)
                # model(input.cuda())
                output = F.softmax(model(npy.cuda()), dim=1)
            softmax = output.cpu().numpy()
        return softmax

    row, col, envs = input_data.shape
    #pre_label = np.zeros((row,col),dtype=np.float32)
    out_result = parallelseg(input_data)
    out_result = out_result.reshape((row, col, 2))

    out_result = out_result[:,:,1]
    out_result = out_result*10000
    return out_result.astype(np.int16)


if __name__ == '__main__':
    # Example usage
    dp = DataPreparation(
        label_path='../dataset/forest_label/Hansen_1km_th30.tif',
        input_parameter_dir='../dataset/environmental',
    )
    data, geot, proj = dp.preprocess_data()

    model_path = './runs/Fold_3_0402-2024-04-02-22-48-07/model_best_total_acc.pth.tar'
    result = go(data, model_path, batch=1000000)
    output_dir='../result/TGS_score/TGS_score.tif'
    write_tiff(output_dir, result, geot, proj)


    
    







