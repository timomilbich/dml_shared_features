################### LIBRARIES ###################
import warnings
warnings.filterwarnings("ignore")

import os, sys, numpy as np, argparse, imp, datetime, time, pickle as pkl, random, json
# os.chdir('/home/karsten_dl/Dropbox/Projects/Confusezius_git/MetricLearning_With_LS-Sep')
# os.chdir('/media/karsten_dl/QS/Data/Dropbox/Projects/Confusezius_git/MetricLearning_With_LS-Sep')
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
import pandas as pd

import torch, torch.nn as nn
import auxiliaries as aux
import datasets as data

# sys.path.insert(0,os.getcwd()+'/models')
from netlib import NetworkSuperClass, NetworkSuperClass_baseline
import netlib
import losses as losses



# ################### INPUT ARGUMENTS ###################
parser = argparse.ArgumentParser()

####### Main Parameter: Dataset to use for Training
parser.add_argument('--dataset',      default='cars196',   type=str, help='Dataset to use.')

### General Training Parameters
parser.add_argument('--lr',                default=0.00001,     type=float, help='Learning Rate for network parameters.')
parser.add_argument('--n_epochs',          default=150,          type=int,   help='Number of training epochs.')
parser.add_argument('--kernels',           default=12,           type=int,   help='Number of workers for pytorch dataloader.')
parser.add_argument('--bs',                default=112 ,        type=int,   help='Mini-Batchsize to use.')
parser.add_argument('--samples_per_class', default=4,           type=int,   help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')
parser.add_argument('--seed',              default=23,           type=int,   help='Random seed for reproducibility.')
parser.add_argument('--scheduler',         default='step',      type=str,   help='Type of learning rate scheduling. Currently: step & exp.')
parser.add_argument('--gamma',             default=0.3,         type=float, help='Learning rate reduction after tau epochs.')
parser.add_argument('--decay',             default=0.0004,      type=float, help='Weight decay for optimizer.')
parser.add_argument('--tau',               default=[10000],nargs='+',type=int,help='Stepsize before reducing learning rate.')

### Class Embedding Settings
parser.add_argument('--classembed',         default=128,          type=int,   help='Embedding Dimension for Class Embedding.')
parser.add_argument('--class_loss',         default='marginloss', type=str,   help='Choose between TripletLoss, ProxyNCA, ...')
parser.add_argument('--class_sampling',     default='distance',   type=str,   help='For triplet-based losses: Modes of Sampling: random, semihard, distance.')
# proxyNCA
parser.add_argument('--class_proxy_lr',     default=0.00001,      type=float, help='PROXYNCA: Learning Rate for Proxies in ProxyNCALoss.')
# margin loss
parser.add_argument('--class_margin',       default=0.2,          type=float, help='TRIPLET, MARGIN: Margin for Triplet Loss')
parser.add_argument('--class_beta_lr',      default=0.0005,       type=float, help='MARGIN: Learning Rate for class margin parameters in MarginLoss')
parser.add_argument('--class_beta',         default=1.2,          type=float, help='MARGIN: Initial Class Margin Parameter in Margin Loss')
parser.add_argument('--class_nu',           default=0,            type=float, help='MARGIN: Regularisation value on betas in Margin Loss.')
parser.add_argument('--class_beta_constant',action='store_true',              help='MARGIN: Keep Beta fixed.')
# multisimilarity loss
parser.add_argument('--class_msim_pos_weight', default=2, type=float, help='Weighting on positive similarities.')
parser.add_argument('--class_msim_neg_weight', default=40, type=float, help='Weighting on negative similarities.')
parser.add_argument('--class_msim_margin', default=0.1, type=float, help='Distance margin for both positive and negative similarities.')
parser.add_argument('--class_msim_pos_thresh', default=0.5, type=float, help='Exponential thresholding.')
parser.add_argument('--class_msim_neg_thresh', default=0.5, type=float, help='Exponential thresholding.')
parser.add_argument('--class_msim_base_mode', default=1, type=int, help='Exponential thresholding.')
parser.add_argument('--class_msim_d_mode', default='cosine', type=str, help='Exponential thresholding.')

### Evaluation Parameters
parser.add_argument('--k_vals',        nargs='+', default=[1,2,4,8], type=int, help='Recall @ Values.')
parser.add_argument('--not_spliteval', action='store_true',                    help='If set, evaluation is done with both embeddings.')

### Network parameters
parser.add_argument('--arch',           default='resnet50',  type=str,    help='Choice of architecture. Limited only to the options avail. to the pretrainedmodels-library.')
parser.add_argument('--not_pretrained', action ='store_true',             help='If set, no pretraining is used for initialization.')
parser.add_argument('--shared_norm',   action='store_true',             help='If set, no weights are saved during training.')

### Setup Parameters
parser.add_argument('--log_online',   action='store_true',             help='If set, no weights are saved during training.')
parser.add_argument('--group',     default='default',          type=str,   help='Appendix to save folder name if any special information is to be included. Will also override the time appendix.')
parser.add_argument('--project',     default='default',          type=str,   help='Appendix to save folder name if any special information is to be included. Will also override the time appendix.')
parser.add_argument('--wandb_key', default='8388187e7c47589ca2875e4007015c7536aede7f', type=str, help='Options are currently: wandb & comet')

parser.add_argument('--gpu',          default=0,           type=int,   help='Random seed for reproducibility.')
parser.add_argument('--no_weights',   action='store_true',             help='If set, no weights are saved during training.')
parser.add_argument('--savename',     default='',          type=str,   help='Appendix to save folder name if any special information is to be included. Will also override the time appendix.')

parser.add_argument('--freq_eval_res',     default=1,    type=int,   help='Embedding Dimension for IntraClass Embedding.')
parser.add_argument('--freq_eval_trainvalset',     default=999,    type=int,   help='Embedding Dimension for IntraClass Embedding.')
parser.add_argument('--freq_analyse_inter_intra_dist',     default=999,    type=int,   help='Embedding Dimension for IntraClass Embedding.')
parser.add_argument('--freq_tsne',     default=999,    type=int,   help='Embedding Dimension for IntraClass Embedding.')

### Paths to datasets and storage folder
parser.add_argument('--source_path',  default='/export/home/karoth/Datasets', type=str,         help='Path to folder containing the dataset folders.')
parser.add_argument('--save_path',    default=os.getcwd()+'/Training_Results', type=str, help='Where to save everything.')

### Read in parameters
opt = parser.parse_args()


# class opt_class(object):
#
#     def __init__(self):
#         self.dataset = 'cars196' # cars196   cub200
#         self.lr = 0.00001
#         self.n_epochs = 150
#         self.kernels = 8
#         self.bs = 128
#         self.samples_per_class = 4
#         self.seed = 23
#         self.scheduler = 'step'
#         self.gamma = 0.3
#         self.decay = 0.0004
#         self.tau = [70, 90] # [70, 90, 120]    [55, 80]
#         self.classembed = 128
#         self.class_loss = 'marginloss'
#         self.class_sampling = 'distance'
#         self.class_proxy_lr = 0.00001
#         self.class_margin = 0.2
#         self.class_beta_lr = 0.0005
#         self.class_beta_constant = False
#         self.class_beta = 1.2
#         self.class_nu = 0
#         self.use_super = False
#         self.not_pretrained = False
#         self.not_spliteval = False
#         self.k_vals = [1,2,4,8]
#         self.arch = 'resnet50'
#         self.no_weights = False
#         self.source_path = '/export/home/karoth/Datasets'
#
#         self.save_path = os.getcwd()+'/Training_Results'
#         self.freq_eval_res = 1
#         self.freq_eval_trainvalset = 25
#         self.freq_tsne = 999
#         self.freq_analyse_inter_intra_dist = 999
#
#         self.gpu = 3
#         self.savename = 'cars196_margin_baseline' # 'baseline_cars_marginloss'
#
# opt = opt_class()

opt.embed_dim = opt.classembed

### If wandb-logging is turned on, initialize the wandb-run here:
if opt.log_online:
    import wandb
    _ = os.system('wandb login {}'.format(opt.wandb_key))
    os.environ['WANDB_API_KEY'] = opt.wandb_key
    opt.unique_run_id = wandb.util.generate_id()
    # wandb.init(id=opt.unique_run_id, resume='allow', project=opt.project, group=opt.group, name=opt.savename, dir=opt.source_path)
    wandb.init(id=opt.unique_run_id, resume='allow', project=opt.project, group=opt.group, name=opt.savename, dir=opt.save_path)
    wandb.config.update(opt)
    # wandb.config.update(opt, allow_val_change=True)

"""============================================================================"""
opt.source_path += '/'+opt.dataset
opt.save_path   += '/'+opt.dataset
print(' **** PROJECT NAME: {} ****'.format(opt.savename))


if opt.dataset=='online_products':
    opt.k_vals = [1,10,100,1000]
if opt.dataset=='in-shop':
    opt.k_vals = [1,10,20,30,50]
if opt.dataset=='vehicle_id':
    opt.k_vals = [1,5]

opt.spliteval = not opt.not_spliteval

if opt.class_loss == 'proxynca': opt.samples_per_class = 1



"""============================================================================"""
################### GPU SETTINGS ###########################
os.environ["CUDA_DEVICE_ORDER"]   ="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(opt.gpu)



"""============================================================================"""
#################### SEEDS FOR REPROD. #####################
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)
torch.backends.cudnn.deterministic=True
torch.manual_seed(opt.seed)
rng = np.random.RandomState(opt.seed)


"""============================================================================"""
##################### NETWORK SETUP ##################
#NOTE: Networks that can be used: 'bninception, resnet50, resnet101, alexnet...'
#>>>>  see import pretrainedmodels; pretrainedmodels.model_names
opt.device    = torch.device('cuda')

model         = NetworkSuperClass_baseline(opt)

opt.input_space, opt.input_range, opt.mean, opt.std = model.input_space, model.input_range, model.mean, model.std
print('{} Setup for {} with {} sampling on {} complete with #weights: {}'.format(opt.class_loss.upper(), opt.arch.upper(), opt.class_sampling.upper()
                                                                                       ,opt.dataset.upper(), aux.gimme_params(model)))

_             = model.to(opt.device)
to_optim   = [{'params':model.parameters(),'lr':opt.lr, 'weight_decay':opt.decay}]



"""============================================================================"""
#################### DATALOADER SETUPS ##################
if opt.dataset in ['cars196', 'cub200', 'online_products']:
    if opt.dataset=='cars196':          train_dataset, val_dataset, train_eval_dataset = data.give_CARS196_datasets(opt)
    if opt.dataset=='cub200':           train_dataset, val_dataset, train_eval_dataset = data.give_CUB200_datasets(opt)
    if opt.dataset=='online_products':  train_dataset, super_train_dataset, val_dataset, train_eval_dataset = data.give_OnlineProducts_datasets(opt)
    train_dataloader        = torch.utils.data.DataLoader(train_dataset,       batch_size=opt.bs, num_workers=opt.kernels, shuffle=True, pin_memory=True, drop_last=True)
    train_eval_dataloader   = torch.utils.data.DataLoader(train_eval_dataset,  batch_size=opt.bs, num_workers=opt.kernels, shuffle=False,pin_memory=True)
    val_dataloader          = torch.utils.data.DataLoader(val_dataset,         batch_size=opt.bs, num_workers=opt.kernels, shuffle=False,pin_memory=True)
    if opt.dataset=='online_products':
        super_train_eval_dataloader = torch.utils.data.DataLoader(super_train_dataset, batch_size=opt.bs, num_workers=opt.kernels, shuffle=False, pin_memory=True)
elif opt.dataset == 'in-shop':
    train_dataset, train_eval_dataset, query_dataset, gallery_dataset = data.give_InShop_datasets(opt)
    train_dataloader    = torch.utils.data.DataLoader(train_dataset,            batch_size=opt.bs, num_workers=opt.kernels, shuffle=True, pin_memory=True, drop_last=True)
    train_eval_dataloader   = torch.utils.data.DataLoader(train_eval_dataset,   batch_size=opt.bs, num_workers=opt.kernels, shuffle=False,pin_memory=True)
    query_dataloader    = torch.utils.data.DataLoader(query_dataset,            batch_size=opt.bs, num_workers=opt.kernels, shuffle=False,pin_memory=True)
    gallery_dataloader  = torch.utils.data.DataLoader(gallery_dataset,          batch_size=opt.bs, num_workers=opt.kernels, shuffle=False,pin_memory=True)
elif opt.dataset == 'vehicle_id':
    train_dataset, train_eval_dataset, small_test_dataset, medium_test_dataset, big_test_dataset = data.give_VehicleID_datasets(opt)
    train_dataloader       = torch.utils.data.DataLoader(train_dataset,         batch_size=opt.bs, num_workers=opt.kernels, shuffle=True, pin_memory=True, drop_last=True)
    train_eval_dataloader   = torch.utils.data.DataLoader(train_eval_dataset,   batch_size=opt.bs, num_workers=opt.kernels, shuffle=False,pin_memory=True)
    small_test_dataloader  = torch.utils.data.DataLoader(small_test_dataset,    batch_size=opt.bs, num_workers=opt.kernels, shuffle=False,pin_memory=True)
    medium_test_dataloader = torch.utils.data.DataLoader(medium_test_dataset,   batch_size=opt.bs, num_workers=opt.kernels, shuffle=False,pin_memory=True)
    big_test_dataloader    = torch.utils.data.DataLoader(big_test_dataset,      batch_size=opt.bs, num_workers=opt.kernels, shuffle=False,pin_memory=True)
else:
    raise Exception('No dataset with name >{}< available!'.format(opt.dataset))


opt.num_classes  = len(train_dataset.avail_classes)




"""============================================================================"""
#################### CREATE LOGGING FILES ###############
aux.set_logging(opt)
InfoPlotter   = aux.InfoPlotter(opt.save_path+'/InfoPlot.svg')
CSV_log_train = aux.CSV_Writer(opt.save_path +'/log_epoch_train.csv', ['Epoch', 'Loss', 'Time'])
if opt.dataset!='vehicle_id':
    CSV_log_val   = aux.CSV_Writer(opt.save_path +'/log_epoch_val.csv', ['Epoch', 'NMI', 'Val Recall Sum'] + ['Recall @ {}'.format(k_val) for k_val in opt.k_vals] + ['Time'])
    CSV_log_trainval   = aux.CSV_Writer(opt.save_path +'/log_epoch_trainval.csv', ['Epoch', 'NMI', 'Val Recall Sum'] + ['Recall @ {}'.format(k_val) for k_val in opt.k_vals] + ['Time'])

    Progress_Saver= {'Train Loss':[], 'Val NMI':[], 'Val Recall Sum':[]}
    for k_val in opt.k_vals: Progress_Saver['Recall @ {}'.format(k_val)] = []
else:
    CSV_log_val   = aux.CSV_Writer(opt.save_path +'/log_epoch_val.csv',   ['Epoch'] + ['Small NMI', 'Medium NMI', 'Big NMI'] +['Small Recall @ 1'.format(k_val) for k_val in opt.k_vals] + \
                                                                          ['Medium Recall @ 1'.format(k_val) for k_val in opt.k_vals] + ['Big Recall @ 1'.format(k_val) for k_val in opt.k_vals] + ['Time'])
    Progress_Saver= {'Train Loss':[]}
    for dtp in ['Small', 'Medium', 'Big']:
        for k_val in opt.k_vals: Progress_Saver[dtp+' '+'Recall @ {}'.format(k_val)] = []
        Progress_Saver[dtp+' NMI'] = []


# save config to project folder
GREEN = '\033[92m'
ENDC = '\033[0m'
print(GREEN + '*************** START TRAINING *******************')
config_logger_text = open('{0}/config_log.txt'.format(opt.save_path), 'a')
vars_params = vars(opt)
for key, value in vars_params.items():
    print('{}: {}'.format(key, value))
    config_logger_text.write('%s %s \n' % (str(key), str(value)))

config_logger_text.close()
print('**************************************************' + ENDC)


"""============================================================================"""
#################### LOSS SETUP ####################
loss_pars = argparse.Namespace()

loss_pars.nu, loss_pars.beta, loss_pars.beta_lr, loss_pars.beta_constant, loss_pars.embed_dim   = opt.class_nu, opt.class_beta, opt.class_beta_lr, opt.class_beta_constant, opt.embed_dim
loss_pars.margin, loss_pars.loss, loss_pars.sampling, loss_pars.num_classes, loss_pars.proxy_lr = opt.class_margin, opt.class_loss, opt.class_sampling, opt.num_classes, opt.class_proxy_lr
loss_pars.class_msim_pos_weight, loss_pars.class_msim_neg_weight, loss_pars.class_msim_margin, loss_pars.class_msim_pos_thresh, loss_pars.class_msim_neg_thresh, loss_pars.class_msim_d_mode, loss_pars.class_msim_base_mode = opt.class_msim_pos_weight, opt.class_msim_neg_weight, opt.class_msim_margin, opt.class_msim_pos_thresh, opt.class_msim_neg_thresh, opt.class_msim_d_mode, opt.class_msim_base_mode
class_criterion, to_optim = losses.loss_select(opt.class_loss, loss_pars, to_optim)

_ = class_criterion.to(opt.device)


"""============================================================================"""
#################### OPTIM SETUP ####################
optimizer  = torch.optim.Adam(to_optim)

if opt.scheduler  =='exp':
    scheduler    = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.gamma)
elif opt.scheduler=='step':
    scheduler    = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.tau, gamma=opt.gamma)
elif opt.scheduler=='cosine':
    optimizer  = torch.optim.Adam(to_optim,   weight_decay=opt.decay)
    opt.T_mul, opt.T_max, opt.T_switch = 2, 10, 10
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.T_max, eta_min=5e-8)
elif opt.scheduler=='none':
    print('Not using any scheduling!')
else:
    raise Exception('No scheduling option for input: {}'.format(opt.scheduler))

global best_recall_score_class
best_recall_score_class = 0


"""============================================================================"""
#################### TRAINER FUNCTION ############################
def train_one_epoch(train_dataloader, model, optimizer, class_criterion, opt, progress_saver, epoch):
    loss_collect_class = []

    start = time.time()
    data_iterator = tqdm(train_dataloader, desc='Epoch {} Training...'.format(epoch), total=len(train_dataloader))
    for i,(class_labels, input) in enumerate(data_iterator):

        #### Train InterClass Embedding
        features  = model(input.to(opt.device))
        loss, sampled_triplets     = class_criterion(features, class_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_collect_class.append(loss.item())

        #####
        if i==len(train_dataloader)-1: data_iterator.set_description('Epoch (Train) {0}: Mean Loss Class [{1:.4f}] ]'.format(
            epoch, np.mean(loss_collect_class)))

    # update logging
    CSV_log_train.log([epoch, np.mean(loss_collect_class), np.round(time.time()-start,4)])
    progress_saver['Train Loss'].append(np.mean(loss_collect_class))
    if opt.log_online:
        wandb.log({f'loss (class)': np.mean(loss_collect_class)}, step=epoch)

"""============================================================================"""
#################### EVALUATION PROTOCOLS ############################
##### For every dataset besides PKU Vehicle-ID and InShop
def evaluate_standard(val_dataloader, model, optimizer, opt, progress_saver, epoch, data_type='val'):
    global best_recall_score_class
    start = time.time()

    with torch.no_grad():
        if opt.dataset!='in-shop':

            # eval class embedding
            NMI, recall_at_ks, feature_coll, image_paths, mean_intra_dist, mean_inter_dist = aux.eval_metrics(model, val_dataloader, opt.device, spliteval=opt.spliteval, epoch=epoch, k_vals = opt.k_vals, opt=opt, embed_type='class')
            if opt.log_online:
                wandb.log({f'nmi (class, {data_type})': NMI}, step=epoch)
                for i, recall in enumerate(recall_at_ks):
                    wandb.log({f'e_recall@{opt.k_vals[i]} (class, {data_type})': recall}, step=epoch)

        # log eval class embedding
        if data_type == 'val':

            # save checkpoint
            if recall_at_ks[0] > best_recall_score_class:
                print('*** !SAVE! Eval (class, val): {} [top: {}]'.format(round(recall_at_ks[0], 4), round(best_recall_score_class, 4)))
                aux.set_checkpoint(model, opt, epoch, optimizer, opt.save_path, progress_saver, suffix='class')
                best_recall_score_class = recall_at_ks[0]

                if opt.dataset != 'in-shop':
                    aux.recover_closest(feature_coll, image_paths, opt.save_path + '/best_test_recovered.png')
                else:
                    aux.recover_closest_inshop(feature_coll, image_paths, opt.save_path + '/best_test_recovered.png')

            if opt.log_online:
                wandb.log({f'best_recall@1 (class)': best_recall_score_class}, step=epoch)

            CSV_log_val.log([epoch, NMI, np.sum(recall_at_ks)] + recall_at_ks + [np.round(time.time()-start)])
            progress_saver['Val NMI'].append(NMI)
            progress_saver['Val Recall Sum'].append(np.sum(recall_at_ks))
            for k_val, recall_val in zip(opt.k_vals, recall_at_ks):
                progress_saver['Recall @ {}'.format(k_val)].append(recall_val)

        # log trainval set
        elif data_type == 'trainval':
            CSV_log_trainval.log([epoch, NMI, np.sum(recall_at_ks)] + recall_at_ks + [np.round(time.time()-start)])

    return NMI, recall_at_ks

##### Evaluation for PKU Vehicle-ID
def evaluate_vehicle_id(small_test_dataloader, medium_test_dataloader, big_test_dataloadaer, model, optimizer, opt, progress_saver, epoch):
    global best_recall_score
    loss_collect = []

    start = time.time()

    with torch.no_grad():
        NMI_small, recall_at_ks_small, feature_coll_small, image_paths_small      = aux.eval_metrics(model, small_test_dataloader, opt.device, epoch=epoch, k_vals = opt.k_vals, opt=opt, desc='Computing Evaluation Metrics - Small Set')
        NMI_medium, recall_at_ks_medium, feature_coll_medium, image_paths_medium  = aux.eval_metrics(model, medium_test_dataloader, opt.device, epoch=epoch, k_vals = opt.k_vals, opt=opt, desc='Computing Evaluation Metrics - Medium Set')
        NMI_big, recall_at_ks_big, feature_coll_big, image_paths_big              = aux.eval_metrics(model, big_test_dataloader, opt.device, epoch=epoch, k_vals = opt.k_vals, opt=opt, desc='Computing Evaluation Metrics - Big Set')

        if recall_at_ks_big[0]>best_recall_score:
            if not opt.no_weights: aux.set_checkpoint(model, opt, epoch,  optimizer, opt.save_path, progress_saver)
            best_recall_score = recall_at_ks_big[0]
            aux.recover_closest(feature_coll_big, image_paths_big, opt.save_path+'/best_test_recovered.png')

        CSV_log_val.log([epoch] + [NMI_small]+[NMI_medium]+[NMI_big]+recall_at_ks_small+recall_at_ks_medium+recall_at_ks_big + [np.round(time.time()-start)])
        for dtp, recall_at_ks, NMI in zip(['Small', 'Medium', 'Big'],[recall_at_ks_small, recall_at_ks_medium, recall_at_ks_big],[NMI_small, NMI_medium, NMI_big]):
            for k_val, recall_val in zip(opt.k_vals,recall_at_ks):
                progress_saver[dtp+' '+'Recall @ {}'.format(k_val)].append(recall_val)
            progress_saver[dtp+' NMI'].append(NMI)

    return [NMI_small]+[NMI_medium]+[NMI_medium], recall_at_ks_small+recall_at_ks_medium+recall_at_ks_big

##### Evaluation for InShop
def evaluate_inshop(query_dataloader, gallery_dataloader, model, optimizer, opt, progress_saver, epoch):
    global best_recall_score
    loss_collect = []

    start = time.time()

    with torch.no_grad():
        NMI, recall_at_ks, query_feature_coll, gallery_feature_coll, query_image_paths, gallery_image_paths = aux.eval_metrics_inshop(model, query_dataloader, gallery_dataloader, opt.device, epoch=epoch, k_vals = opt.k_vals, opt=opt)

        if recall_at_ks[0]>best_recall_score:
            if not opt.no_weights: aux.set_checkpoint(model, opt, epoch,  optimizer, opt.save_path, progress_saver)
            best_recall_score = recall_at_ks[0]
            aux.recover_closest_inshop(query_feature_coll, gallery_feature_coll, query_image_paths, gallery_image_paths, opt.save_path+'/best_test_recovered.png')

        CSV_log_val.log([epoch, NMI, np.sum(recall_at_ks)] + recall_at_ks + [np.round(time.time()-start)])
        progress_saver['Val NMI'].append(NMI)
        progress_saver['Val Recall Sum'].append(np.sum(recall_at_ks))
        for k_val, recall_val in zip(opt.k_vals,recall_at_ks):
            progress_saver['Recall @ {}'.format(k_val)].append(recall_val)

    return NMI, recall_at_ks



"""============================================================================"""


#################### MAIN PART ############################
print('\n-----\n')
for epoch in range(opt.n_epochs):
    print('Model: {}'.format(opt.savename))

    if opt.scheduler=='cosine' and epoch==opt.T_switch:
        opt.T_max    *= opt.T_mul
        opt.T_switch += opt.T_max
        cluster_labels = losses.cluster(opt, train_eval_dataloader, model, mode=opt.mode_main, num_cluster=opt.num_cluster, full=opt.full_embed_cluster, ignore_last=opt.ignore_last)
        scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.T_max, eta_min=5e-8)
    if opt.scheduler!='none': print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_lr())))

    ### Train one epoch
    model.train()
    train_one_epoch(train_dataloader, model, optimizer, class_criterion, opt, Progress_Saver, epoch)

    ### Evaluate -
    _ = model.eval()
    if opt.dataset in ['cars196', 'cub200', 'online_products']:
        # perfom evaluation on val set
        NMI, recall_at_ks = evaluate_standard(val_dataloader, model, optimizer, opt, Progress_Saver, epoch, data_type='val')

        # perfom evaluation on trainval set
        if epoch % opt.freq_eval_trainvalset == 0 and epoch > 0:
            NMI, recall_at_ks = evaluate_standard(train_eval_dataloader, model, optimizer, opt, Progress_Saver, epoch, data_type='trainval')

        recall_str        = ' '.join('{0:3.3f}@{1}'.format(recall_val, k_val) for recall_val, k_val  in zip([np.max(Progress_Saver['Recall @ {}'.format(k_val)]) for k_val in opt.k_vals], opt.k_vals))
        InfoPlotter.title = 'NMI:{0:2.4f} | Recalls: {1}'.format(np.max(Progress_Saver['Val NMI']), recall_str)
        InfoPlotter.make_plot(range(epoch+1), Progress_Saver['Train Loss'], [Progress_Saver['Recall @ {}'.format(k)] for k in opt.k_vals], ['Train Loss']+['Recall @ {}'.format(k) for k in opt.k_vals])

        ### Create tSNE plots
        # perplex = [10, 30, 50]
        # if epoch % opt.freq_tsne == 0 and epoch > 0:
            # print('*** Creating tSNE plots...')
            # aux.plot_tSNE(opt, train_eval_dataloader, model, feat='embed', type_data='trainval', n_samples=700, img_size=60,
            #               perplex=perplex, img_zoom=0.7, epoch=epoch)
            # aux.plot_tSNE(opt, val_dataloader, model, feat='embed', type_data='val', n_samples=700, img_size=60,
            #               perplex=perplex, img_zoom=0.7, epoch=epoch)
            # print('*** Creating tSNE plots... DONE!')

    elif opt.dataset == 'in-shop':
        NMI, recall_at_ks = evaluate_inshop(query_dataloader, gallery_dataloader, model, optimizer, opt, Progress_Saver, epoch)
        recall_str        = ' '.join('{0:3.3f}@{1}'.format(recall_val, k_val) for recall_val, k_val  in zip([np.max(Progress_Saver['Recall @ {}'.format(k_val)]) for k_val in opt.k_vals], opt.k_vals))
        InfoPlotter.title = 'NMI:{0:2.4f} | Recalls: {1}'.format(np.max(Progress_Saver['Val NMI']), recall_str)
        InfoPlotter.make_plot(range(epoch+1), Progress_Saver['Train Loss'], [Progress_Saver['Recall @ {}'.format(k)] for k in opt.k_vals], ['Train Loss']+['Recall @ {}'.format(k) for k in opt.k_vals])
    elif opt.dataset == 'vehicle_id':
        NMI, recall_at_ks = evaluate_vehicle_id(small_test_dataloader, medium_test_dataloader, big_test_dataloader, model, optimizer, opt, Progress_Saver, epoch)
        vals = [np.max(x) for _,x in Progress_Saver.items()]
        output_str = 'NMIs: {} | Recalls: {}'.format(np.round(vals[:3],4), np.round(vals[3:]))
        InfoPlotter.title = output_str
        recalls = [[Progress_Saver[dtp+' Recall @ {}'.format(k)] for k in opt.k_vals] for dtp in ['Small', 'Medium', 'Big']]
        recalls = [x for y in recalls for x in y]
        titles = [[dtp+' Recall @ {}'.format(k) for k in opt.k_vals] for dtp in ['Small', 'Medium', 'Big']]
        titles = [x for y in titles for x in y]
        InfoPlotter.make_plot(range(epoch+1), Progress_Saver['Train Loss'], recalls, ['Train Loss']+titles)

    if opt.scheduler != 'none':
        scheduler.step()


    print('\n-----\n')
