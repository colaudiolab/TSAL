# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest_feature.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def train(models, train_loader, test_loader,cycle,logger,args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)

    # Initialize all of the statistics we want to keep track of
    loss_meter = AverageMeter()
    global_step = 0
    if not isinstance(models['backbone'], nn.DataParallel):
        models['backbone'] = nn.DataParallel(models['backbone'])
    models['backbone'] = models['backbone'].to(device)
    # diff lr optimizer
    mlp_list = ['mlp_head.0.weight', 'mlp_head.0.bias', 'mlp_head.1.weight', 'mlp_head.1.bias']
    mlp_params = list(filter(lambda kv: kv[0] in mlp_list, models['backbone'].module.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in mlp_list, models['backbone'].module.named_parameters()))
    mlp_params = [i[1] for i in mlp_params]
    base_params = [i[1] for i in base_params]
    # only finetuning small/tiny models on balanced audioset uses different learning rate for mlp head
    optimizer = torch.optim.Adam([{'params': base_params, 'lr': args.lr}, {'params': mlp_params, 'lr': args.lr * args.head_lr}], weight_decay=5e-7, betas=(0.95, 0.999))
    mlp_lr = optimizer.param_groups[1]['lr']
    lr_list = [args.lr, mlp_lr]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
    loss_fn = nn.CrossEntropyLoss()
    args.loss_fn = loss_fn
    models['backbone'].train()
    best_acc = 0.
    for epoch in range(1,args.n_epochs+1):
        for i, (audio_input, labels) in enumerate(train_loader):
            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if global_step <= 1000 and global_step % 50 == 0 and args.warmup == True:
                for group_id, param_group in enumerate(optimizer.param_groups):
                    warm_lr = (global_step / 1000) * lr_list[group_id]
                    param_group['lr'] = warm_lr
            optimizer.zero_grad()
            audio_output,_,_ = models['backbone'](audio_input, args.task)

            loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), B)
            global_step += 1
        if epoch % 5 == 0 or epoch == 1:
            stats, valid_loss = validate(models, test_loader, args)
            mAUC = np.mean([stat['auc'] for stat in stats])
            acc = stats[0]['acc']
            middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
            middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
            average_precision = np.mean(middle_ps)
            average_recall = np.mean(middle_rs)
            if best_acc < acc:
                best_acc = acc
            print(args.dataset, 'Cycle:', cycle + 1, 'Epoch:', epoch, '---',
                  'Val Acc: {:.2f} \t Best Acc: {:.2f} \t Val AUC: {:.2f} \t Avg Precision: {:.2f} \t Avg Recall: {:.2f} \t d_prime: {:.2f} \t train_loss: {:.2f} \t valid_loss: {:.2f}'.
                  format(acc*100, best_acc*100,mAUC*100,average_precision*100,average_recall*100,d_prime(mAUC)*100,loss_meter.avg,valid_loss), flush=True)
            logger.info("Cycle: {}. Epoch:{:.2f}. Val Acc: {:.2f}. Best Acc: {:.2f}. Val AUC: {:.2f}. Avg Precision: {:.2f}. Avg Recall: {:.2f}. d_prime: {:.2f}. train_loss: {:.2f}. valid_loss: {:.2f} ".
                  format(cycle + 1, epoch, acc*100, best_acc*100,mAUC*100,average_precision*100,average_recall*100,d_prime(mAUC)*100,loss_meter.avg,valid_loss))
            models['backbone'].train()
        scheduler.step()
        loss_meter.reset()


def validate(models, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(models['backbone'], nn.DataParallel):
        models['backbone'] = nn.DataParallel(models['backbone'])
    models['backbone'] = models['backbone'].to(device)
    # switch to evaluate mode
    models['backbone'].eval()

    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(val_loader):
            audio_input = audio_input.to(device)

            # compute output
            audio_output,_,_ = models['backbone'](audio_input, args.task)
            audio_output = torch.sigmoid(audio_output)
            predictions = audio_output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)

            # compute the loss
            labels = labels.to(device)
            if isinstance(args.loss_fn, torch.nn.CrossEntropyLoss):
                loss = args.loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            else:
                loss = args.loss_fn(audio_output, labels)
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        stats = calculate_stats(audio_output, target)

    return stats, loss

