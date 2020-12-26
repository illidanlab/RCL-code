# -*- coding:utf-8 -*-
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
from model import CNN
import argparse, sys
import numpy as np
import datetime
import shutil
import pickle as pkl

from loss import loss_multi_consensus

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.5)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, or cifar100', default = 'cifar10')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--fuzzy_exponent', type=float, default=1.0, help='exponent for the knowledge share fuzzy rate')
parser.add_argument('--stop_fuzzy', type=int, default=2, help='when to stop fuzzy compared to sample decay')
parser.add_argument('--num_networks', type=int, default=2, help='number of network branches')

args = parser.parse_args()

# Seed
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

# Hyper Parameters
batch_size = 128
learning_rate = args.lr 

# load dataset
if args.dataset=='mnist':
    input_channel=1
    num_classes=10
    args.top_bn = False
    args.epoch_decay_start = 80
    args.n_epoch = 200
    train_dataset = MNIST(root='./data/',
                                download=True,  
                                train=True, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                         )
    
    test_dataset = MNIST(root='./data/',
                               download=True,  
                               train=False, 
                               transform=transforms.ToTensor(),
                               noise_type=args.noise_type,
                               noise_rate=args.noise_rate
                        )
    
if args.dataset=='cifar10':
    input_channel=3
    num_classes=10
    args.top_bn = False
    args.epoch_decay_start = 80
    args.n_epoch = 200
    train_dataset = CIFAR10(root='./data/',
                                download=True,  
                                train=True, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                           )
    
    test_dataset = CIFAR10(root='./data/',
                                download=True,  
                                train=False, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                          )

if args.dataset=='cifar100':
    input_channel=3
    num_classes=100
    args.top_bn = False
    args.epoch_decay_start = 100
    args.n_epoch = 200                   # change for test
    if args.noise_type == 'pairflip':
        args.num_gradual = 15
    train_dataset = CIFAR100(root='./data/',
                                download=True,  
                                train=True, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                            )
    
    test_dataset = CIFAR100(root='./data/',
                                download=True,  
                                train=False, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                            )

if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate

noise_or_not = train_dataset.noise_or_not

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1
        
# define drop rate schedule
rate_schedule = np.ones(args.n_epoch)*forget_rate
rate_schedule[1:args.num_gradual+1] = np.linspace(0, forget_rate**args.exponent, args.num_gradual)
print(rate_schedule)

## fix fuzzy exponent in regular experiment
#if args.noise_type=='symmetric':
#    args.fuzzy_exponent = 0.5
#elif args.noise_type=='pairflip':
#    args.fuzzy_exponent = 2.0

knowledge_fuzzy_rate = np.zeros(args.n_epoch)
knowledge_fuzzy_rate[1:args.num_gradual*args.stop_fuzzy+1] = 1 - (np.arange(0, args.num_gradual*args.stop_fuzzy, 1)/float(args.stop_fuzzy*args.num_gradual))**args.fuzzy_exponent
print(knowledge_fuzzy_rate)

save_dir = args.result_dir +'/' +args.dataset+'/multi_css/'

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str=args.dataset+'_multi_css_'+args.noise_type+'_'+str(args.noise_rate)+'_net_'+str(args.num_networks) + '_seed_' + str(args.seed)

txtfile=save_dir+"/"+model_str+".txt"
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model
def train(train_loader,epoch, all_models, all_optimizers):
    print 'Training %s...' % model_str
    pure_ratio_list=[]
    css_ratio_list=[]

    train_total=0
    train_correct=0
    for i, (images, labels, indexes) in enumerate(train_loader):
        # print('iteration===={:d}'.format(i))
        # print(indexes)

        ind=indexes.cpu().numpy().transpose()
        if i>args.num_iter_per_epoch:
            break
      
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        
        # Forward + Backward + Optimize
        all_logits = dict()
        for k in range(args.num_networks):
            model_name = 'model' + str(k)
            logits_name = 'logits' + str(k)

            model = all_models[model_name]
            logits = model(images)
            all_logits[logits_name] = logits

            if k==0:
                prec, _ = accuracy(logits, labels, topk=(1, 5))
                train_total+=1
                train_correct+=prec

        all_losses, all_pure_ratios, all_css_ratios = loss_multi_consensus(all_logits, labels, rate_schedule[epoch], ind, noise_or_not, knowledge_fuzzy_rate[epoch], i)
        pure_ratio_list.append(100*all_pure_ratios[0])
        css_ratio_list.append(100*all_css_ratios[0])

        for k in range(args.num_networks):
            optimizer_name = 'optimizer' + str(k)
            optimizer = all_optimizers[optimizer_name]
            loss_name = 'loss' + str(k)
            loss = all_losses[loss_name]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Loss1: %.4f, Pure Ratio1: %.4f, Consensus Ratio1: %.4f '
                  %(epoch, args.n_epoch, i+1, len(train_dataset)//batch_size, prec, all_losses['loss0'].data, np.sum(pure_ratio_list)/len(pure_ratio_list), np.sum(css_ratio_list)/len(css_ratio_list)))

    train_acc=float(train_correct)/float(train_total)
    return train_acc, pure_ratio_list, css_ratio_list

# Evaluate the Model
def evaluate(test_loader, all_models):
    #print 'Evaluating %s...' % model_str

    all_accs = []
    for k in range(args.num_networks):
        model_name = 'model' + str(k)
        model = all_models[model_name]

        model.eval()
        correct = 0
        total = 0
        for images, labels, _ in test_loader:
            images = Variable(images).cuda()
            logits = model(images)
            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred.cpu() == labels).sum()
        acc = 100*float(correct)/float(total)
        all_accs.append(acc)

    return all_accs


def main():
    seed_everything(args.seed)
    # Data Loader (Input Pipeline)
    print 'loading dataset...'
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
    # Define models
    print 'building model...'
    all_models = dict()
    all_optimizers = dict()
    for k in range(args.num_networks):
        model_name = 'model' + str(k)
        optimizer_name = 'optimizer' + str(k)

        model = CNN(input_channel=input_channel, n_outputs=num_classes)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if k==0:
            print(model.parameters)

        all_models[model_name] = model
        all_optimizers[optimizer_name] = optimizer

    mean_pure_ratio1=0
    mean_css_ratio1=0
    with open(txtfile, "a") as myfile:
        myfile.write('epoch: train_acc1 test_acc1 pure_ratio1 css_ratio1\n')

    epoch=0
    train_acc1=0
    # evaluate models with random weights
    test_accs=evaluate(test_loader, all_models)
    print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Pure Ratio1 %.4f %% CSS Ratio1 %.4f %%' % (epoch, args.n_epoch, len(test_dataset), test_accs[0], mean_pure_ratio1, mean_css_ratio1))
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ': '  + str(train_acc1) + ' '  +  str(test_accs[0]) + ' ' + str(mean_pure_ratio1)+ ' ' + str(mean_css_ratio1) + "\n")

    # training
    for epoch in range(1, args.n_epoch):

        for k in range(args.num_networks):
            model_name = 'model' + str(k)
            optimizer_name = 'optimizer' + str(k)

            model = all_models[model_name]
            optimizer = all_optimizers[optimizer_name]

            model.train()
            adjust_learning_rate(optimizer, epoch)

        # train models
        train_acc1, pure_ratio_1_list, css_ratio_1_list = train(train_loader, epoch, all_models, all_optimizers)
        # evaluate models
        test_accs = evaluate(test_loader, all_models)
        # save results
        mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
        mean_css_ratio1 = sum(css_ratio_1_list)/len(css_ratio_1_list)
        print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %%, Pure Ratio 1 %.4f %% CSS Ratio 1 %.4f %%' % (epoch, args.n_epoch, len(test_dataset), test_accs[0], mean_pure_ratio1, mean_css_ratio1))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': ' + str(train_acc1) + ' ' + str(test_accs[0]) + ' ' + str(mean_pure_ratio1)+ ' ' + str(mean_css_ratio1) + "\n")


if __name__=='__main__':
    main()
