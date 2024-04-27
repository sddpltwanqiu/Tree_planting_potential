
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from uitls.utils import net
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from tensorboardX import SummaryWriter
from TGS_score_model.TGS_trainer import SetDataset, ToTensor , AverageMeter
ss = StandardScaler()
import torchvision.transforms as transforms

def test(test_loader, model, criterion):
    model.eval()
    losses = AverageMeter()
    f1 = AverageMeter()
    prec = AverageMeter()
    recall = AverageMeter()
    total_acc = AverageMeter()

    for i, samples in enumerate(test_loader):
        target = samples["y"]
        target = target.reshape(-1, 1)
        target = target.cuda()
        input = samples["x"]
        input = input.reshape(-1, 18)
        input = input.cuda()

        with torch.no_grad():
            inputvar = torch.autograd.Variable(input)
            targetvar = torch.autograd.Variable(target)

            out = model(inputvar)
            loss = criterion(out, targetvar.squeeze())

        prediction = torch.argmax(out, 1)
        preds = prediction.data.cpu().numpy()
        gts = target.cpu().numpy().reshape(-1)

        losses.update(loss.item(), input.size(0))
        acc = accuracy_score(gts, preds)
        total_acc.update(acc, input.size(0))
        f1_val = f1_score(gts, preds, average='macro')
        f1.update(f1_val, input.size(0))
        prec_val = precision_score(gts, preds, average='macro')
        prec.update(prec_val, input.size(0))
        recall_val = recall_score(gts, preds, average='macro')
        recall.update(recall_val, input.size(0))

    print('Test Results: Acc {acc.avg:.4f} | F1 {f1.avg:.4f} | Prec {prec.avg:.4f} | Recall {recall.avg:.4f}'
          .format(acc=total_acc, f1=f1, prec=prec, recall=recall))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset', default='../dataset/UseForTraining/test_3.npy', type=str)
    parser.add_argument('--batch_size', default=10000, type=int)
    parser.add_argument('--num_para', default=18, type=int)
    parser.add_argument('--model_path', default='./runs/Fold_3_0402-2024-04-02-22-48-07/model_best_total_acc.pth.tar', type=str)
    args = parser.parse_args()

    model = net.MLPNet(args.num_para, 64, 32, 2)
    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    if os.path.isfile(args.model_path):
        print("=> loading model '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("=> no model found at '{}'".format(args.model_path))
        return

    test_dataset = np.load(args.test_dataset)
    mean = np.mean(test_dataset, axis=0)
    std = np.std(test_dataset, axis=0)
    test_dataset[:, :-1] = (test_dataset[:, :-1] - mean[:-1]) / std[:-1]
    classes = test_dataset.shape[1]

    nums, classes = test_dataset.shape

    # Dynamically compute the number of samples that can be included in each batch
    batch_size = args.batch_size
    test_sample_count = nums
    should_num = (batch_size * np.floor(test_sample_count / batch_size)).astype(np.int32)
    test_dataset = test_dataset[:should_num, :]

    test_dataset = test_dataset.reshape(-1, batch_size, classes)  # 
    
    
    test_dataset = SetDataset(test_dataset, transforms.Compose([ToTensor()]))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()

    test(test_loader, model, criterion)

if __name__ == '__main__':
    main()
