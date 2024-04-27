import argparse
import os
import shutil
import time
from warnings import simplefilter

from uitls.utils import net
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils.loss_new import F1_Loss, Precision_loss
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import StandardScaler
from tensorboardX import SummaryWriter

ss = StandardScaler()
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'

parser = argparse.ArgumentParser()
parser.add_argument('--epoches', default=1000, type=int)
parser.add_argument('--read_batch_size', default=4, type=int)
parser.add_argument('--batch_size', default=1000000, type=int)
parser.add_argument('--print_freq', default=1, type=int)
parser.add_argument('--num_para', default=18, type=int)
parser.add_argument('--para_train_path', default='', type=str)
parser.add_argument('--para_test_path', default='', type=str)
parser.add_argument('--para_valid_path', default='', type=str)
parser.add_argument('--tensorboard', action='store_true')
parser.add_argument('--input_path', default='../dataset/UseForTraning/', type=str)
parser.add_argument('--input_dataset', default='All_trees.npy', type=str)
parser.add_argument('--expertiment_name',
                    default="All_trees_0408", type=str)

parser.set_defaults(tensorboard=True)
best_prec1 = 0
args = parser.parse_args()
name = args.expertiment_name + \
    time.strftime("-%Y-%m-%d-%H-%M-%S", time.localtime())

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

def main():
    global args, best_prec1, writer
    if args.tensorboard:
        writer = SummaryWriter()

    inpthfile = args.input_path + '//' + args.input_dataset

    train_dataset = np.load(inpthfile)
    mean = np.mean(train_dataset, axis=0)
    std = np.std(train_dataset, axis=0)

    train_dataset[:, :-1] = (train_dataset[:, :-1] - mean[:-1]) / std[:-1]

    nums, classes = train_dataset.shape
    # should_num = (args.batch_size * np.floor(nums /
    #                                          args.batch_size)).astype(np.int32)
    # train_dataset = train_dataset[:should_num, :]

    valid_num = np.floor(nums*0.25).astype(np.int32)
    train_num = (nums - valid_num).astype(np.int32)
    train_dataset = train_dataset[valid_num:, :]
    should_num = (args.batch_size * np.floor(train_num /
                                            args.batch_size)).astype(np.int32)
    #
    train_dataset = train_dataset.reshape(-1, args.batch_size, classes)

    val_dataset = train_dataset[:valid_num, :]
    should_num2 = (args.batch_size * np.floor(valid_num /
                                              args.batch_size)).astype(np.int32)
    val_dataset = val_dataset[:should_num2, :]
    val_dataset = val_dataset.reshape(-1, args.batch_size, classes)

    transform = transforms.Compose([ToTensor()])

    train_dataset = SetDataset(train_dataset, transform)
    #test_dataset = SetDataset(test_dataset,transform)
    val_dataset = SetDataset(val_dataset, transform)

    kwargs = {"num_workers": 1000, "pin_memory": True}

    train_loader = torch.utils.data.DataLoader(
        train_dataset, args.read_batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, args.read_batch_size, shuffle=True)

    cudnn.benchmark = True

    model = net.MLPNet(args.num_para, 64, 32, 2)
    nGPUs = torch.cuda.device_count()
    if nGPUs >= 1:
        model = model.cuda()
        model = nn.DataParallel(model)

    optim = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()  # F1_Loss()  # Precision_loss()

    best_f1 = 0
    best_prec = 0
    best_recall = 0
    n_best_f1 = 0
    n_best_prec = 0
    n_best_recall = 0
    best_total_acc = 0
    for epoch in range(args.epoches):
        train(train_loader, model, criterion, optim, epoch)

        total_acc, pre_f1, best_prec, pre_recall, n_best_prec, n_pre_recall, n_pre_f1 = validate(
            val_loader, model, criterion, epoch)

        is_f1_best = pre_f1 > best_f1
        is_prec_best = best_prec > best_prec
        is_recall_best = pre_recall > best_recall
        is_n_f1_best = n_pre_f1 > n_best_f1
        is_n_prec_best = n_best_prec > n_best_prec
        is_n_recall_best = n_pre_recall > n_best_recall

        is_best_total_acc = total_acc > best_total_acc

        best_total_acc = max(best_total_acc, total_acc)
        best_f1 = max(pre_f1, best_f1)
        best_prec = max(best_prec, best_prec)
        best_recall = max(pre_recall, best_recall)
        n_best_f1 = max(n_pre_f1, n_best_f1)
        n_best_prec = max(best_prec, n_best_prec)
        n_best_recall = max(n_pre_recall, n_best_recall)

        save_checkpoint({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'best_f1': pre_f1,
            'best_accuracy': best_total_acc,
            'best_precision': best_prec,
            'best_recall': pre_recall,
            'n_best_f1': n_pre_f1,
            'n_best_accuracy': n_best_prec,
            'n_best_recall': n_pre_recall
        }, is_f1_best, is_prec_best, is_recall_best, is_n_recall_best, is_n_prec_best, is_n_f1_best, is_best_total_acc)

    print('Best F1-socre:', best_f1)
    writer.close()


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()

    f1 = AverageMeter()
    prec = AverageMeter()
    recall = AverageMeter()

    n_f1 = AverageMeter()
    n_prec = AverageMeter()
    n_recall = AverageMeter()

    total_acc = AverageMeter()

    model.train()

    end = time.time()
    for i, samples in enumerate(train_loader):
        target = samples["y"]
        target = target.reshape(-1, 1)
        target = target.cuda()
        input = samples["x"]
        input = input.reshape(-1, 18)
        input = input.cuda()

        inputvar = torch.autograd.Variable(input)
        targetvar = torch.autograd.Variable(target)

        out = model(inputvar)

        loss = criterion(out, targetvar.squeeze())

        gts, preds = [], []
        n_gts, n_preds = [], []
        prediction = torch.argmax(out, 1)
        pred = prediction.data.cpu().numpy()
        gt = target.cpu().numpy()
        for pred_, gt_ in zip(pred, gt):
            preds.append(pred_)
            gts.append(gt_)
            n_preds.append(1-pred_)
            n_gts.append(1-gt_)

        losses.update(loss.item(), input.size(0))

        prec1 = accuracy(preds, gts, n_class=out.size(1))
        f1.update(prec1['f1-score'], input.size(0))
        prec.update(prec1['precision'], input.size(0))
        recall.update(prec1['recall'], input.size(0))

        n_prec1 = accuracy(n_preds, n_gts, n_class=out.size(1))
        n_f1.update(n_prec1['f1-score'], input.size(0))
        n_recall.update(n_prec1['recall'], input.size(0))
        n_prec.update(n_prec1['precision'], input.size(0))

        total_acc.update(prec1['acc'], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('EPOCH:[{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuary {acc.val:.4f} ({acc.avg:.4f})\t'
                  'F1-score {f1.val:.4f} ({f1.avg:.4f}); {n_f1.val:.4f} ({n_f1.avg:.4f})\t'
                  'Precision {prec.val:.4f} ({prec.avg:.4f}; {n_prec.val:.4f} ({n_prec.avg:.4f})\t'
                  'Recall {recall.val:.4f} ({recall.avg:.4f}; {n_recall.val:.4f} ({n_recall.avg:.4f})'
                  .format(
                      epoch, i, len(train_loader), loss=losses, acc=total_acc, f1=f1, n_f1=n_f1, prec=prec, n_prec=n_prec, recall=recall, n_recall=n_recall
                  ))

    if args.tensorboard:
        writer.add_scalar('train_loss', losses.avg, epoch)
        writer.add_scalar('train_f1', f1.avg, epoch)
        writer.add_scalar('train_precision', prec.avg, epoch)
        writer.add_scalar('train_recall', recall.avg, epoch)
        writer.add_scalar('train_n_f1', n_f1.avg, epoch)
        writer.add_scalar('train_n_precision', n_prec.avg, epoch)
        writer.add_scalar('train_n_recall', n_recall.avg, epoch)
        writer.add_scalar('train_accuary', total_acc.avg, epoch)


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    f1 = AverageMeter()
    prec = AverageMeter()
    recall = AverageMeter()

    n_f1 = AverageMeter()
    n_prec = AverageMeter()
    n_recall = AverageMeter()

    total_acc = AverageMeter()

    model.eval()

    end = time.time()
    for i, samples in enumerate(val_loader):
        target = samples["y"]
        target = target.reshape(-1, 1)
        target = target.cuda()
        input = samples["x"]
        input = input.reshape(-1, 18)
        input = input.cuda()

        inputvar = torch.autograd.Variable(input)
        targetvar = torch.autograd.Variable(target)

        with torch.no_grad():
            out = model(inputvar)
            loss = criterion(out, targetvar.squeeze())

        gts, preds = [], []
        n_gts, n_preds = [], []
        prediction = torch.argmax(out, 1)
        pred = prediction.data.cpu().numpy()
        gt = target.cpu().numpy()
        for pred_, gt_ in zip(pred, gt):
            preds.append(pred_)
            gts.append(gt_)
            n_preds.append(1-pred_)
            n_gts.append(1-gt_)

        losses.update(loss.item(), input.size(0))

        prec1 = accuracy(preds, gts, n_class=out.size(1))
        f1.update(prec1['f1-score'], input.size(0))
        prec.update(prec1['precision'], input.size(0))
        recall.update(prec1['recall'], input.size(0))

        n_prec1 = accuracy(n_preds, n_gts, n_class=out.size(1))
        n_f1.update(n_prec1['f1-score'], input.size(0))
        n_recall.update(n_prec1['recall'], input.size(0))
        n_prec.update(n_prec1['precision'], input.size(0))

        total_acc.update(prec1['acc'], input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('TEST:[{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuary {acc.val:.4f} ({acc.avg:.4f})\t'
                  'F1-score {f1.val:.4f} ({f1.avg:.4f}); {n_f1.val:.4f} ({n_f1.avg:.4f})\t'
                  'Precision {prec.val:.4f} ({prec.avg:.4f}; {n_prec.val:.4f} ({n_prec.avg:.4f})\t'
                  'Recall {recall.val:.4f} ({recall.avg:.4f}; {n_recall.val:.4f} ({n_recall.avg:.4f})'
                  .format(
                      epoch, i, len(val_loader), loss=losses, acc=total_acc, f1=f1, n_f1=n_f1, prec=prec, n_prec=n_prec, recall=recall, n_recall=n_recall
                  ))
    print('*Acc {acc.avg:.4f}\t'
          'F1-score {f1.avg:.4f};{n_f1.avg:.4f}\t'
          'Precision {prec.avg:.4f};{n_prec.avg:.4f}\t'
          'Recall {recall.avg:.4f};{n_recall.avg:.4f}'
          .format(f1=f1, n_f1=n_f1, acc=total_acc, prec=prec, n_prec=n_prec, recall=recall, n_recall=n_recall))

    if args.tensorboard:
        writer.add_scalar('val_loss', losses.avg, epoch)
        writer.add_scalar('val_f1', f1.avg, epoch)
        writer.add_scalar('val_prec', prec.avg, epoch)
        writer.add_scalar('val_recall', recall.avg, epoch)
        writer.add_scalar('val_n_f1', n_f1.avg, epoch)
        writer.add_scalar('val_n_prec', n_prec.avg, epoch)
        writer.add_scalar('val_n_recall', n_recall.avg, epoch)
        writer.add_scalar('val_accuary', total_acc.avg, epoch)
    return total_acc.avg, f1.avg, prec.avg, recall.avg, n_prec.avg, n_recall.avg, n_f1.avg


def accuracy(label_true, label_pred, n_class):
    f1 = f1_score(label_true, label_pred)
    acc = accuracy_score(label_true, label_pred)
    recall = recall_score(label_true, label_pred)
    precision = precision_score(label_true, label_pred)
    return {"acc": acc,
            "f1-score": f1,
            "recall": recall,
            "precision": precision}


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum/self.count


class ToTensor(object):

    def __init__(self):
        super(ToTensor, self).__init__()

    def __call__(self, sample):
        x, y = sample["x"], sample["y"]
        x = x.astype(np.float32)
        x = ss.fit_transform(x)
        return {
            "x": torch.from_numpy(x).type(torch.FloatTensor),
            "y": torch.from_numpy(y).type(torch.LongTensor)
        }


class SetDataset(torch.utils.data.Dataset):

    def __init__(self, location, tranform=None):
        self.location = location
        self.transform = transforms

    def __len__(self):
        return len(self.location)

    def __getitem__(self, idx):
        x = self.location[idx, :, :-1]
        y = self.location[idx, :, -1]

        sample = {
            "x": torch.FloatTensor(x),
            "y": torch.LongTensor([y])
        }

        return sample


def save_checkpoint(state, is_f1_best, is_prec_best, is_recall_best, is_n_recall_best, is_n_prec_best, is_n_f1_best, is_best_total_acc, filename='checkpoint.pth.tar'):
    global name
    dirt = "runs/{}/".format(name)
    if not os.path.exists(dirt):
        os.makedirs(dirt)
    filename = dirt + filename
    torch.save(state, filename)
    if is_f1_best:
        shutil.copyfile(filename, 'runs/{}/'.format(name) +
                        "model_best_f1.pth.tar")
    if is_prec_best:
        shutil.copyfile(filename, 'runs/{}/'.format(name) +
                        "model_best_prec.pth.tar")
    if is_recall_best:
        shutil.copyfile(filename, 'runs/{}/'.format(name) +
                        "model_best_recall.pth.tar")
    if is_n_f1_best:
        shutil.copyfile(filename, 'runs/{}/'.format(name) +
                        "model_best_n_f1.pth.tar")
    if is_n_prec_best:
        shutil.copyfile(filename, 'runs/{}/'.format(name) +
                        "model_best_n_prec.pth.tar")
    if is_n_recall_best:
        shutil.copyfile(filename, 'runs/{}/'.format(name) +
                        "model_best_n_recall.pth.tar")
    if is_best_total_acc:
        shutil.copyfile(filename, 'runs/{}/'.format(name) +
                        "model_best_total_acc.pth.tar")


if __name__ == "__main__":
    main()
