import torch
from torch import nn


class MLPNet(nn.Module):
    def __init__(self, input_size,hiddensize1, hiddensize2,num_classes):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hiddensize1)   
        self.BN1 = nn.BatchNorm1d(hiddensize1)
        self.fc2 = nn.Linear(hiddensize1, hiddensize2)
        self.BN2 = nn.BatchNorm1d(hiddensize2)
        self.fc3 = nn.Linear(hiddensize2, num_classes)
        
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self,x):
        out = self.fc1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


class MLPNet1(nn.Module):
    def __init__(self, input_size,hiddensize1, num_classes):
        super(MLPNet1, self).__init__()
        self.fc1 = nn.Linear(input_size, hiddensize1)   
        self.BN1 = nn.BatchNorm1d(hiddensize1)
        self.fc2 = nn.Linear(hiddensize1, num_classes)
    
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self,x):
        out = self.fc1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
     

class MLPNet2(nn.Module):
    def __init__(self, input_size,num_classes):
        super(MLPNet2, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)   
        #self.BN1 = nn.BatchNorm1d(hiddensize1)
        self.relu = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self,x):
        out = self.fc1(x)
        #out = self.BN1(out)
        out = self.relu(out)
        return out

