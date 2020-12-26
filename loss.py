import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from functools import reduce
from sklearn.metrics import confusion_matrix


def loss_spl(y, t, forget_rate, ind, noise_or_not):
    include_or_not = np.zeros([len(y), 1], dtype=np.int8)

    loss = F.cross_entropy(y, t, reduce=False)
    ind_sorted = np.argsort(loss.cpu().data).cuda()  
    loss_sorted = loss[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))
    # print('remember rate {:.3f} num_remember {:d} batch_size {:d}'.format(remember_rate, num_remember, len(loss_sorted)))

    pure_ratio = np.sum(noise_or_not[ind[ind_sorted.cpu()[:num_remember]]]) / float(num_remember)  # cpu
    ind_update = ind_sorted[:num_remember]
    loss_update = F.cross_entropy(y[ind_update], t[ind_update])

    include_or_not[ind_update.cpu()] = 1
    return loss_update, pure_ratio, include_or_not        # change to newest


def update_index(ind_union, ind_intersect, fuzzy_rate, seed=1):
    np.random.seed(seed)
    n_intersect = len(ind_intersect)
    n_union = len(ind_union)

    if n_intersect == n_union:
        ind_update = ind_intersect
    else:
        n_in = int(np.floor((n_union-n_intersect)*fuzzy_rate))
        ind_diff = np.setdiff1d(ind_union, ind_intersect)
        ind_in = np.random.choice(ind_diff, n_in)
        ind_update = np.concatenate([ind_intersect, ind_in])
    return ind_update


def loss_multi_consensus(all_logits, t, forget_rate, ind, noise_or_not, fuzzy_rate, iter):

    n_nets = len(all_logits)
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(t))

    all_inds = dict()
    all_pure_ratios = np.zeros(n_nets)
    for i in range(n_nets):
        logits_name = 'logits' + str(i)
        logits = all_logits[logits_name]

        loss = F.cross_entropy(logits, t, reduce=False)     #reduce will be decreped, use reduction='none' instead
        ind_sorted = np.argsort(loss.cpu().data).cuda()

        all_pure_ratios[i] = np.sum(noise_or_not[ind[ind_sorted.cpu()[:num_remember]]]) / float(num_remember)  # cpu

        ind_name = 'ind' + str(i)
        all_inds[ind_name] = ind_sorted[:num_remember]

    all_losses_update = dict()  # for storing loss as torch varibales
    all_consensus_ratios = []
    for i in range(n_nets):
        loss_name = 'loss' + str(i)
        # current ind_update from all the rest inds
        ind_rest = []
        for k in range(n_nets):
            if k != i:
                ind_rest.append(all_inds['ind'+str(k)].cpu().data)
        # print(ind_rest)

        ind_union = reduce(np.union1d, ind_rest)
        ind_intersect = reduce(np.intersect1d, ind_rest)
        all_consensus_ratios.append(len(ind_intersect)/float(num_remember))

        ind_ensemble = update_index(ind_union, ind_intersect, fuzzy_rate)

        all_losses_update[loss_name] = F.cross_entropy(all_logits['logits'+str(i)][ind_ensemble], t[ind_ensemble]) # /len(ind_ensemble)

        if iter==0 & i==0:
            print(iter, i)
            n_clean = np.sum(noise_or_not[ind[ind_ensemble]])
            print('remember rate {:.3f} remember # {:d} include # {:d} clean # {:d} pr {:.4f}'.format(remember_rate, num_remember, len(ind_ensemble), n_clean, n_clean/float(len(ind_ensemble))))
    return all_losses_update, all_pure_ratios, all_consensus_ratios

