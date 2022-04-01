from __future__ import print_function
import argparse
import os
import time
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn import metrics
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p
from Model_SeNet import se_resnet50
from Model_RegNet import RegnetY_400MF, RegnetY_800MF
import NewResNet_Model
import torchvision.models as models
import dataset
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='ccRCC develop and validate')
# Datasets
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')                  
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='val batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 45],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0,1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#Selfsup_or_not
parser.add_argument('--selfsup', default=0, type=int,
                    help='Use selfsup pretrain or not')
parser.add_argument('--model_choose', default=0, type=int,
                    help='0-se 1-res101 2-reg400 3-reg800 ')
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best val accuracy
best_epoch = 0
best_t5 = 0.
best_auc = 0
best_auc_top = 0

def main():
    global best_acc, best_epoch, best_t5, best_auc, best_auc_top

    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    transform_train = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5329571, 0.5329571, 0.5329571), (0.2350305, 0.2350305, 0.2350305)),   
        Resize(224),
    ])
    transform_selftrain = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5329571, 0.5329571, 0.5329571), (0.2350305, 0.2350305, 0.2350305)),   
    Resize(224),
])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5454319, 0.5454319, 0.5454319), (0.23387246, 0.23387246, 0.23387246)), 
        Resize(224),
    ])

    self_pretrainset = dataset.self_supCCR(root='xxxxxxxxxxxxx',train=True, transform=transform_selftrain)
    self_pretestset = dataset.self_supCCR(root='xxxxxxxxxxxxx',train=False, transform=transform_test)

    trainset = dataset.Timing_CCR(root='xxxxxxxxxxxxx',train=True, transform=transform_train)
    testset = dataset.Timing_CCR(root='xxxxxxxxxxxxx',train=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    self_pretrain_loader = torch.utils.data.DataLoader(
        self_pretrainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    val_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    self_preval_loader = torch.utils.data.DataLoader(
        self_pretestset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    print("==> creating model")
    if args.selfsup:
        if args.model_choose == 0:
            model = se_resnet50(num_classes=4, selfsup=1)
        elif args.model_choose == 1:
            model = models.resnet101(num_classes=4)
        elif args.model_choose == 2:
            model = RegnetY_400MF(sup_or_not=1)
        elif args.model_choose == 3:
            model = RegnetY_800MF(sup_or_not=1)  
    else:
        # convert the classifier into nonlinear projection
        if args.model_choose == 0:
            model = se_resnet50(selfsup=0, num_classes=2)
        elif args.model_choose == 1:
            model = NewResNet_Model.resnet101(num_classes=2)
        elif args.model_choose == 2:
            model = RegnetY_400MF(sup_or_not=0)
        elif args.model_choose == 3:
            model = RegnetY_800MF(sup_or_not=0)

    if not args.selfsup:
        model_dict = model.state_dict()
        if args.model_choose == 0:
            pretrained_dict = torch.load("xxxxxxxxxxxxx")['state_dict']
        elif args.model_choose == 1:
            pretrained_dict = torch.load("xxxxxxxxxxxxx")['state_dict']
        elif args.model_choose == 2:
            pretrained_dict = torch.load("xxxxxxxxxxxxx")['state_dict']
        elif args.model_choose == 3:
            pretrained_dict = torch.load("xxxxxxxxxxxxx")['state_dict']
        pretrained_dict = {k[7:]: v for k, v in list(pretrained_dict.items())[:-2]}    #only projection
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        if args.model_choose == 2 or args.model_choose == 3:
            model.net[-1].fc1.apply(weight_init_kaiming)
            model.net[-1].fc2.apply(weight_init_kaiming)
            model.net[-1].fc3.apply(weight_init_kaiming)
            
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    if not args.selfsup:
        weight = torch.tensor([1.,2.7])  
        weight = weight.cuda()
        criterion = nn.CrossEntropyLoss(weight=weight)  #sample reweighting

        # the projection is pretrained
        projection = list(model.parameters())[:-2]
        classification = list(model.parameters())[-2:]
        optimizer_pro = optim.SGD(projection, lr=args.lr * 0.1, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True) 
        optimizer_cla = optim.SGD(classification, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        optimizer = [optimizer_pro, optimizer_cla]
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'cifar-10-'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Accuracy', 'Sensitivity', 'Specificity', 'AUC_top', 'CutOFF'])

     # develop and val
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        # selfsup
        if args.selfsup:
            adjust_learning_rate(optimizer, epoch)      
            train_loss, train_acc = train(self_pretrain_loader, model, criterion, optimizer, epoch, use_cuda, args.selfsup)
            (val_loss, val_acc, 
            acc, optimal_sensitivity, optimal_specificity, Auc, optimal_cutoff) = test(self_preval_loader, model, criterion, epoch, use_cuda, args.selfsup)
        else: 
            #developing
            adjust_learning_rate_warmup(optimizer_pro, optimizer_cla, epoch)  
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda, args.selfsup)
            (val_loss, val_acc, 
            acc, optimal_sensitivity, optimal_specificity, Auc, optimal_cutoff) = test(val_loader, model, criterion, epoch, use_cuda, args.selfsup)
        logger.append([state['lr'], train_loss, val_loss, train_acc, val_acc, acc, optimal_sensitivity, optimal_specificity, Auc, optimal_cutoff])
        
        # save model
        if args.selfsup:
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            if is_best == True:
                best_epoch = epoch + 1
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': val_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint=args.checkpoint)
            print('Best Acc: {}     Best epoch: {} '.format(best_acc, best_epoch))
        else:
            is_best_top = Auc > best_auc_top
            best_auc_top = max(Auc, best_auc_top)
            if is_best_top == True:
                best_epoch_top = epoch + 1
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': val_acc,
                    'best_acc': best_acc,
                    'optimizer0' : optimizer[0].state_dict(),
                    'optimizer1' : optimizer[1].state_dict(),
                }, is_best_top, checkpoint=args.checkpoint)
            print('Best Auc: {}   Best epoch: {}'.format(best_auc_top , best_epoch_top))
    if args.selfsup:
        logger.append([state['lr'], 0., best_epoch, best_acc, 0, 0, 0, 0, 0, 0])
    else:
        logger.append([state['lr'], 0., 0, 0, 0, 0, best_epoch_top, best_auc_top, 0, 0])
    logger.plot()
    logger.close()

def train(trainloader, model, criterion, optimizer, epoch, use_cuda, selfsup):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))

    for batch_idx, (inputs, targets, patient) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)

        # mixed loss strategy
        if not selfsup:
            new_target = torch.zeros_like(targets).cuda()
            loss = 0.6 * criterion(outputs, targets) + 0.4 * criterion(outputs, new_target)     
        else:
            loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data)
        losses.update(loss.item(), targets.size(0))
        top1.update(prec1[0].item(), targets.size(0))

        # compute gradient and do SGD step
        if selfsup:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        else:
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            loss.backward()
            optimizer[0].step()
            optimizer[1].step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


@torch.no_grad()
def test(testloader, model, criterion, epoch, use_cuda, selfsup):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    if not selfsup:
        statistics = statistics_by_patient()
        statistics.reset()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets, patient) in enumerate(testloader):

        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)

        # Counting
        outputs_prob = nn.Softmax(dim=-1)(outputs)
        
        if not selfsup:
            for i in range(inputs.size(0)):
                statistics.update(patient[i].item(), outputs_prob[i][1].detach().cpu().numpy(), targets[i].detach().cpu())

        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data)
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0].item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg
                    )
        bar.next()
    bar.finish()
    if not selfsup:
        (acc, optimal_sensitivity, optimal_specificity, Auc, optimal_cutoff) = statistics._get_acc_by_maximum_n_mvisocre_new(1)
        return (losses.avg, top1.avg, acc, optimal_sensitivity, optimal_specificity, Auc, optimal_cutoff)
    else:
        return (losses.avg, top1.avg, 0, 0, 0, 0, 0)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath) 
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def weight_init_kaiming(m):
    class_names = m.__class__.__name__
    if class_names.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif class_names.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data)


#cosine learning rate 
def adjust_learning_rate_warmup(optimizer_pro, optimizer_cla, epoch):
    import math
    global state
    if epoch < 5:
        lr = args.lr * (epoch+1) / 5.
    else:
        lr = args.lr * 0.5 * (1 + math.cos((epoch+1) * math.pi / args.epochs))
    for param_group in optimizer_pro.param_groups:
        param_group['lr'] = lr * 0.1
    for param_group in optimizer_cla.param_groups:
        param_group['lr'] = lr
    state['lr'] = lr


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


class statistics_by_patient():
    def __init__(self, infopath='./'):
        self.infopath = infopath

    def reset(self):
        self.final = {}
        self.patient_list = []
        self.root = 'xxxxxxxxxxxxx'
        with open('%s/xxxxxxxxxxxxx'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()
                    if float(entry[1].replace('-', '.')) not in self.patient_list:
                        self.patient_list.append(float(entry[1].replace('-', '.')))
        
        for patient in self.patient_list:
            self.final[patient] = {'prob':[], 'label':[]}

    def update(self, patient_i, prob_i, label_i):
        self.final[patient_i]['prob'].append(prob_i)
        self.final[patient_i]['label'].append(label_i)

    def _get_acc_by_maximum_n_mvisocre_new(self, n):
        predict = []
        predict_mean = []
        label = []
        top_k = n 
        for patient_i in self.patient_list:
            # check
            check_set = set([i.item() for i in self.final[patient_i]['label']])
            if len(check_set) != 1:
                print("    Wrong when processing {} | label incorrect".format(patient_i))
                print("    Elements:", check_set)
            predict_mean.append(np.mean(self.final[patient_i]['prob']))
            top_k_idx = np.array(self.final[patient_i]['prob']).argsort()[-top_k:]
            top_k_val = np.array(self.final[patient_i]['prob'])[top_k_idx]
            predict_val = np.mean(top_k_val)
            predict.append(predict_val)
            label.append(self.final[patient_i]['label'][0])

        fpr, tpr, threshold = metrics.roc_curve(label, predict)
        Auc = metrics.auc(fpr, tpr) 

        # find optimal cut-off value using Youden_index
        y = tpr - fpr
        youden_index = np.argmax(y)
        optimal_cutoff = threshold[youden_index]
        optimal_sensitivity = tpr[youden_index]
        optimal_specificity = 1 - fpr[youden_index]

        # compute acc
        tp = optimal_sensitivity * 38    
        tn = optimal_specificity * 76   
        acc = (tp+tn)/114                
        
        return (acc, optimal_sensitivity, optimal_specificity, Auc, optimal_cutoff)


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img = img.unsqueeze(0)
        final_img = F.interpolate(img, size=(self.size, self.size), mode="bilinear")
        final_img = torch.squeeze(final_img)
        return final_img


class Padding(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        c = img.size(0)
        h = img.size(1)
        w = img.size(2)

        if h > self.size or w >self.size:                     
            img = img.unsqueeze(0)
            final_img = torch.nn.functional.interpolate(img, size=(self.size, self.size), mode="bilinear")
            final_img = torch.squeeze(final_img)
        else:
            final_img = torch.zeros((c, self.size, self.size))

            half = self.size // 2
            half_h_edge = int(round(h / 2, 0))  
            half_w_edge = int(round(w / 2, 0))
            dest_hs = half - half_h_edge
            dest_he = dest_hs + h
            dest_ws = half - half_w_edge
            dest_we = dest_ws + w

            final_img[:, dest_hs:dest_he, dest_ws:dest_we] = img
        return final_img

if __name__ == '__main__':
    main()