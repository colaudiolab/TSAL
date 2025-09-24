# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
import ast
import random
import sys

import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import SubsetRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
from models import ASTModel
import numpy as np
from traintest_feature import train
import logging
import sys
from datetime import datetime
import torch.nn.functional as F


torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'
    return datetime.today().strftime(fmt)


def setup_default_logging(args, default_level=logging.INFO,
                          format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s"):
    output_dir1 = os.path.join(args.dataset, args.sampling)
    output_dir = os.path.join(output_dir1, str(args.fold))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger('train')

    logging.basicConfig(  # unlike the root logger, a custom logger can’t be configured using basicConfig()
        filename=os.path.join(output_dir, f'{time_str()}_{args.top_p_percent_hard}_{args.lambda_db}.log'),
        format=format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=default_level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(default_level)
    console_handler.setFormatter(logging.Formatter(format))
    logger.addHandler(console_handler)

    return logger, output_dir


def min_max_normalize(scores):
    scores = np.array(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    return (scores - min_score) / (max_score - min_score)


def get_gradients(model, inputs, device):
    model.eval()
    inputs.requires_grad_(True)
    logits, _, _ = model(inputs,args.task)
    predicted_labels = torch.argmax(logits, dim=1)
    one_hot_targets = torch.zeros_like(logits).scatter_(1, predicted_labels.unsqueeze(1), 1).to(device)
    loss = (logits * one_hot_targets).sum()
    model.zero_grad()
    loss.backward(retain_graph=True)
    grads = inputs.grad.data
    inputs.requires_grad_(False)
    return grads

def get_hard_examples_dynamic(model, dataloaders, device,hard_pool_percentage):
    model.eval()
    all_losses = []
    feature_embeddings=[]
    all_labeled_grads=[]
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            logits, feature_embedding, _ = model(inputs, args.task)
            loss = F.cross_entropy(logits, labels, reduction='none')
            feature_embedding = feature_embedding.mean(dim=1)
        feature_embeddings.append(feature_embedding.cpu().numpy())
        all_losses.append(loss.cpu().numpy())
        grads_batch = get_gradients(model, inputs, device)
        all_labeled_grads.append(grads_batch.cpu().numpy())
    feature_embeddings = np.vstack(feature_embeddings)
    all_labeled_grads = np.concatenate(all_labeled_grads)
    all_losses = np.concatenate(all_losses)
    avg_loss = np.mean(all_losses)
    H_dynamic_indices = [idx for idx, l in enumerate(all_losses) if l > avg_loss]
    loss_lists = np.array(all_losses)[H_dynamic_indices]
    normalized_losses = min_max_normalize(loss_lists)
    scores = np.array(normalized_losses)
    probabilities = scores / np.sum(scores)
    num_to_select = int(len(H_dynamic_indices) * hard_pool_percentage)
    H_representative_indices = H_dynamic_indices

    H_representative_indices = np.random.choice(
         H_dynamic_indices, size=num_to_select, replace=False, p=probabilities
    ).tolist()

    feature_embeddings = np.array(feature_embeddings)[H_representative_indices]
    all_labeled_grads = np.array(all_labeled_grads)[H_representative_indices]
    return feature_embeddings,all_labeled_grads

def get_uncertainty(model, dataloaders, device,lambda_db,hard_pool_percentage):
    feature_embeddiongs,all_labeled_grads = get_hard_examples_dynamic(model, dataloaders, device,hard_pool_percentage)
    model.eval()
    unlabel_embeddings = []
    unlabel_losses = []
    for inputs_u, _ in dataloaders['extra']:
        inputs_u = inputs_u.to(device)
        with torch.no_grad():
            _, embeddings_u, _ = model(inputs_u, args.task)
            embeddings_u = embeddings_u.mean(dim=1)
        grads_u = get_gradients(model, inputs_u, device)
        unlabel_embeddings.append(embeddings_u.cpu().numpy())
        unlabel_losses.append(grads_u.cpu().numpy())
    unlabel_embeddings = np.vstack(unlabel_embeddings)
    unlabel_losses = np.concatenate(unlabel_losses)
    emb_sim = cosine_similarity(unlabel_embeddings, feature_embeddiongs)
    unlabel_losses = unlabel_losses.reshape(unlabel_losses.shape[0], -1)
    all_labeled_grads = all_labeled_grads.reshape(all_labeled_grads.shape[0], -1)

    gd_metric = cosine_similarity(unlabel_losses,all_labeled_grads)
    query_scores = emb_sim+lambda_db*gd_metric
    hard_proximity_scores = np.max(query_scores, axis=1)
    return hard_proximity_scores



print("Training------")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default=None, help="training data json")
parser.add_argument("--data-val", type=str, default=None, help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")

parser.add_argument("--dataset", type=str, help="the dataset used for training")
parser.add_argument("--dataset_mean", type=float, help="the dataset mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, help="the dataset std, used for input normalization")
parser.add_argument("--target_length", type=int, help="the input length in frames")
parser.add_argument("--num_mel_bins", type=int, default=128, help="number of input mel bins")

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=16, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# only used in pretraining stage or from-scratch fine-tuning experiments
parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument('--adaptschedule', help='if use adaptive scheduler ', type=ast.literal_eval, default='False')

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the models or not', type=ast.literal_eval)

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
# the stride used in patch spliting, e.g., for patch size 16*16, a stride of 16 means no overlapping, a stride of 10 means overlap of 6.
# during self-supervised pretraining stage, no patch split overlapping is used (to aviod shortcuts), i.e., fstride=fshape and tstride=tshape
# during fine-tuning, using patch split overlapping (i.e., smaller {f,t}stride than {f,t}shape) improves the performance.
# it is OK to use different {f,t} stride in pretraining and finetuning stages (though fstride is better to keep the same)
# but {f,t}stride in pretraining and finetuning stages must be consistent.
parser.add_argument("--fstride", type=int, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument("--fshape", type=int, help="shape of patch on the frequency dimension")
parser.add_argument("--tshape", type=int, help="shape of patch on the time dimension")
parser.add_argument('--model_size', help='the size of AST models', type=str, default='base384')

parser.add_argument("--task", type=str, default='ft_cls', help="pretraining or fine-tuning task", choices=["ft_avgtok", "ft_cls", "pretrain_mpc", "pretrain_mpg", "pretrain_joint"])

# pretraining augments
#parser.add_argument('--pretrain_stage', help='True for self-supervised pretraining stage, False for fine-tuning stage', type=ast.literal_eval, default='False')
parser.add_argument('--mask_patch', help='how many patches to mask (used only for ssl pretraining)', type=int, default=400)
parser.add_argument("--cluster_factor", type=int, default=3, help="mask clutering factor")
parser.add_argument("--epoch_iter", type=int, default=2000, help="for pretraining, how many iterations to verify and save models")

# fine-tuning arguments
parser.add_argument("--pretrained_mdl_path", type=str, default=None, help="the ssl pretrained models path")
parser.add_argument("--head_lr", type=int, default=1, help="the factor of mlp-head_lr/lr, used in some fine-tuning experiments only")
parser.add_argument("--noise", help='if augment noise in finetuning', type=ast.literal_eval)
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=5, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument("--wa", help='if do weight averaging in finetuning', type=ast.literal_eval)
parser.add_argument("--wa_start", type=int, default=16, help="which epoch to start weight averaging in finetuning")
parser.add_argument("--wa_end", type=int, default=30, help="which epoch to end weight averaging in finetuning")
parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["BCE", "CE"])

parser.add_argument('--sampling', default='RANDOM', type=str, help='data sampling method')
parser.add_argument("--cycles", type=int)
parser.add_argument("--fold", type=int)
parser.add_argument("--folds", type=int)
parser.add_argument("--top_p_percent_hard", type=float)
parser.add_argument("--lambda_db", type=float)
args = parser.parse_args()
val_audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                  'mode': 'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False}
audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset,
              'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':args.noise}
# extra_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
#               'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':args.noise}


train_set = dataloader.AudioDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf)
# extra_set = dataloader.AudioDataset(args.data_train, label_csv=args.label_csv, audio_conf=extra_conf)
num_training = len(train_set)
indices = list(range(num_training))
random.shuffle(indices)
start_indices = int(num_training*0.1)
labeled_set = indices[:start_indices]
unlabeled_set = indices[start_indices:]
add_samples = int(num_training * 0.05)
forgetting_scores = np.full(num_training, -np.inf)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                           sampler=SubsetRandomSampler(labeled_set), num_workers=0,
                                           pin_memory=False, drop_last=True)
extra_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                           sampler=SubsetRandomSampler(unlabeled_set), num_workers=0,
                                           pin_memory=False, drop_last=True)
test_loader = torch.utils.data.DataLoader(
    dataloader.AudioDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
    batch_size=args.batch_size * 2, shuffle=False, num_workers=0, pin_memory=False)
audio_model = ASTModel(label_dim=args.n_class, fshape=args.fshape, tshape=args.tshape, fstride=args.fstride, tstride=args.tstride,
                   input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size, pretrain_stage=False,
                   load_pretrained_mdl_path=args.pretrained_mdl_path)


if not isinstance(audio_model, torch.nn.DataParallel):
    audio_model = torch.nn.DataParallel(audio_model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {'backbone': audio_model}
dataloaders = {'train': train_loader, 'test': test_loader, 'extra': extra_loader}
logger, _ = setup_default_logging(args)


for cycle in range(args.cycles):
    train(models, dataloaders['train'], dataloaders['test'],cycle,logger,args)

    random.shuffle(unlabeled_set)
    if args.sampling == 'random':
        subset = unlabeled_set[:add_samples]
        labeled_set += subset
        unlabeled_set = unlabeled_set[add_samples:]
    else:
        print('this is {}'.format(args.sampling))
        uncertainty = get_uncertainty(
            model=models['backbone'],
            dataloaders=dataloaders,
            device=device,
            lambda_db=args.lambda_db,
            hard_pool_percentage=args.top_p_percent_hard
        )
        arg = np.argsort(uncertainty)
        labeled_set += list(torch.tensor(unlabeled_set)[arg][-add_samples:].numpy())
        unlabeled_set = list(torch.tensor(unlabeled_set)[arg][:-add_samples].numpy())
    dataloaders['train'] = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               sampler=SubsetRandomSampler(labeled_set), num_workers=args.num_workers,
                                               pin_memory=False, drop_last=True)
    dataloaders['extra'] = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               sampler=SubsetRandomSampler(unlabeled_set), num_workers=args.num_workers,
                                               pin_memory=False, drop_last=True)

