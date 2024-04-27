import os
import net
import torch
import shutil
import argparse
import numpy as np
import torch.nn.functional as F
from osgeo import gdal
from utils.utils import read_tiff, write_tiff

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'

parser = argparse.ArgumentParser()
parser.add_argument('--num_para', default=18, type=int)
parser.add_argument('--tensorboard', action='store_true')
parser.add_argument('--input_path', default='../dataset_20240402', type=str)
parser.add_argument('--input_dataset', default='All_trees_0408.npy', type=str)


inpth = r'/home/liutang/World_tree_carbon/dataset/Predction/input_submat'
oupth = r'/home/liutang/World_tree_carbon/dataset/Predction/Output_submat'
wholepth = r'/home/liutang/World_tree_carbon/dataset/Predction/Output_wholetif'
model_path = r'/home/liutang/World_tree_carbon/code/inference/best_f1_model'

meanpth = r'/home/liutang/World_tree_carbon/dataset/Predction/input_wholemat/mean'
stdpth = r'/home/liutang/World_tree_carbon/dataset/Predction/input_wholemat/std'

def go(innpy, meannpy, stdnpy, model_path, batch):
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

    input_mat = np.load(innpy)
    mean_mat = np.load(meannpy)
    std_mat = np.load(stdnpy)

    input_data = (input_mat[:,:,:-2]-mean_mat[:-2])/std_mat[:-2]

    row, col, envs = input_data.shape
    #pre_label = np.zeros((row,col),dtype=np.float32)
    out_result = parallelseg(input_data)
    out_result = out_result.reshape((row, col, 2))

    out_result = out_result[:,:,1]
    out_result = out_result*10000
    return out_result.astype(np.int16)

def batch_prec(inpth, oupth, model_path, class_num):

    if not os.path.exists(oupth):
        os.makedirs(oupth)

    meannpy = os.path.join(meanpth, 'array_mean.npy')
    stdnpy = os.path.join(stdpth, 'array_std.npy')

    file_list = os.listdir(inpth)
    for file in file_list:
        in_file_pth = os.path.join(inpth, file)
        oufile = go(in_file_pth, meannpy, stdnpy, model_path, class_num, batch=2200000)
        ou_file_pth = os.path.join(oupth, file)
        np.save(ou_file_pth, oufile)

    ec_pth = r'/home/liutang/World_tree_carbon/dataset/Parameter/EC/BIO_C_2017.tif'


def submat2tif(inpth,outpth,class_num):
    
    ec_label,geot, proj = read_tiff(ec_pth)
    ec_label = ec_label[:, :, 0]
    ec_label[ec_label < 0] = -1
    ec_label = ec_label.astype(np.int8)
    ec_label[ec_label!=class_num] = 0
    ec_label[ec_label==class_num] = 1
    # out_array = out_array * ec_label

    ec_out_pth = os.path.join(r'/home/liutang/World_tree_carbon/dataset/Predction/ec_class_label', class_name + '.tif')

    outpth = os.path.join(outpth, class_name + '.tif')
    #write_tiff(outpth, out_array, geot, proj)
    write_tiff(ec_out_pth, ec_label, geot, proj)


if __name__ == "__main__":
    for c in range(1,5):
        #batch_prec(inpth, oupth, model_path, c)
        submat2tif(oupth, wholepth, c)
    






