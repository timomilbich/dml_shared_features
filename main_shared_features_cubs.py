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
from netlib import NetworkSuperClass
import netlib
import losses as losses



# ################### INPUT ARGUMENTS ###################
# parser = argparse.ArgumentParser()
#
# ####### Main Parameter: Dataset to use for Training
# parser.add_argument('--dataset',      default='cub200',   type=str, help='Dataset to use.')
#
# ### General Training Parameters
# parser.add_argument('--lr',                default=0.00001,     type=float, help='Learning Rate for network parameters.')
# parser.add_argument('--n_epochs',          default=80,          type=int,   help='Number of training epochs.')
# parser.add_argument('--kernels',           default=8,           type=int,   help='Number of workers for pytorch dataloader.')
# parser.add_argument('--bs',                default=112 ,        type=int,   help='Mini-Batchsize to use.')
# parser.add_argument('--samples_per_class', default=4,           type=int,   help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')
# parser.add_argument('--seed',              default=1,           type=int,   help='Random seed for reproducibility.')
# parser.add_argument('--scheduler',         default='step',      type=str,   help='Type of learning rate scheduling. Currently: step & exp.')
# parser.add_argument('--gamma',             default=0.3,         type=float, help='Learning rate reduction after tau epochs.')
# parser.add_argument('--decay',             default=0.0004,      type=float, help='Weight decay for optimizer.')
# parser.add_argument('--tau',               default=[30,55],nargs='+',type=int,help='Stepsize before reducing learning rate.')
#
# ### Class Embedding Settings
# parser.add_argument('--classembed',         default=128,          type=int,   help='Embedding Dimension for Class Embedding.')
# parser.add_argument('--class_loss',         default='marginloss', type=str,   help='Choose between TripletLoss, ProxyNCA, ...')
# parser.add_argument('--class_sampling',     default='distance',   type=str,   help='For triplet-based losses: Modes of Sampling: random, semihard, distance.')
# parser.add_argument('--class_proxy_lr',     default=0.00001,      type=float, help='PROXYNCA: Learning Rate for Proxies in ProxyNCALoss.')
# parser.add_argument('--class_margin',       default=0.2,          type=float, help='TRIPLET, MARGIN: Margin for Triplet Loss')
# parser.add_argument('--class_beta_lr',      default=0.0005,       type=float, help='MARGIN: Learning Rate for class margin parameters in MarginLoss')
# parser.add_argument('--class_beta',         default=1.2,          type=float, help='MARGIN: Initial Class Margin Parameter in Margin Loss')
# parser.add_argument('--class_nu',           default=0,            type=float, help='MARGIN: Regularisation value on betas in Margin Loss.')
# parser.add_argument('--class_beta_constant',action='store_true',              help='MARGIN: Keep Beta fixed.')
#
# ### IntraClass Embedding Settings
# parser.add_argument('--intraclassembed',     default=128,    type=int,   help='Embedding Dimension for IntraClass Embedding.')
# parser.add_argument('--intra_loss',          default='marginloss', type=str, help='Clustering mode: Without normalization (no_norm) or with mean-subtraction (mean), mean-std-norm (mstd) or whitening (white).')
# parser.add_argument('--intra_sampling',      default='distance',   type=str, help='Clustering mode: Without normalization (no_norm) or with mean-subtraction (mean), mean-std-norm (mstd) or whitening (white).')
# parser.add_argument('--intra_proxy_lr',      default=0.00001,      type=float, help='PROXYNCA: Learning Rate for Proxies in ProxyNCALoss.')
# parser.add_argument('--intra_margin',        default=0.2,    type=float, help='Margin value for cluster criterion.')
# parser.add_argument('--intra_beta_lr',       default=0.0005,       type=float, help='MARGIN: Learning Rate for class margin parameters in MarginLoss')
# parser.add_argument('--intra_beta',          default=1.2,          type=float, help='MARGIN: Initial Class Margin Parameter in Margin Loss')
# parser.add_argument('--intra_nu',            default=0,            type=float, help='MARGIN: Regularisation value on betas in Margin Loss.')
# parser.add_argument('--intra_beta_constant', action='store_true',              help='MARGIN: Keep Beta fixed.')
#
# parser.add_argument('--no_adv_class_reverse', action='store_true',              help='MARGIN: Keep Beta fixed.')
# parser.add_argument('--no_adv_intra_reverse', action='store_true',              help='MARGIN: Keep Beta fixed.')
#
# ### MIC Parameters
# parser.add_argument('--disw',                default=0.001,  type=float, help='Weight on adversarial loss.')
# parser.add_argument('--advnet_dim',          default=512,    type=int,   help='Dimensionality of adversarial network.')
# parser.add_argument('--advnet_decay',        default=0,      type=float,   help='Weight decay for adversarial network.')
# parser.add_argument('--num_cluster',         default=30,     type=int,   help='Number of clusters.')
# parser.add_argument('--cluster_update',      default=1,      type=int,   help='Number of epochs to train before updating cluster labels.')
# parser.add_argument('--mode',                default='mstd', type=str,   help='Clustering mode: Without normalization (no_norm) or with mean-subtraction (mean), mean-std-norm (mstd) or whitening (white).')
# parser.add_argument('--use_super',           action ='store_true',        help='use super labels for SOP.')
#
# ### Evaluation Parameters
# parser.add_argument('--k_vals',        nargs='+', default=[1,2,4,8], type=int, help='Recall @ Values.')
# parser.add_argument('--not_spliteval', action='store_true',                    help='If set, evaluation is done with both embeddings.')
#
# ### Network parameters
# parser.add_argument('--arch',           default='resnet50',  type=str,    help='Choice of architecture. Limited only to the options avail. to the pretrainedmodels-library.')
# parser.add_argument('--not_pretrained', action ='store_true',             help='If set, no pretraining is used for initialization.')
#
# ### Setup Parameters
# parser.add_argument('--gpu',          default=0,           type=int,   help='Random seed for reproducibility.')
# parser.add_argument('--no_weights',   action='store_true',             help='If set, no weights are saved during training.')
# parser.add_argument('--savename',     default='',          type=str,   help='Appendix to save folder name if any special information is to be included. Will also override the time appendix.')
#
# ### Paths to datasets and storage folder
# parser.add_argument('--source_path',  default=os.getcwd()+'/Datasets', type=str,         help='Path to folder containing the dataset folders.')
# parser.add_argument('--save_path',    default=os.getcwd()+'/Training_Results', type=str, help='Where to save everything.')
#
# ### Read in parameters
# opt = parser.parse_args()


class opt_class(object):

    def __init__(self):
        self.dataset = 'cub200' # cars196   cub200
        self.shared_norm = False
        self.lr = 0.00001
        self.n_epochs = 150
        self.kernels = 20 #8
        self.bs = 112
        self.samples_per_class = 4
        self.seed = 1
        self.scheduler = 'step'
        self.gamma = 0.3
        self.decay = 0.0004
        self.tau = [55, 80]  # [70, 90]
        self.classembed = 128
        self.class_loss = 'marginloss'
        self.class_sampling = 'distance'
        self.class_proxy_lr = 0.00001
        self.class_margin = 0.2
        self.class_beta_lr = 0.0005
        self.class_beta_constant = False
        self.class_beta = 1.2
        self.class_nu = 0
        self.intraclassembed = 128
        self.intra_loss = 'marginloss_noise' #'marginloss_noise'
        self.intra_sampling = 'distance_noise' # 'distance_noise'
        self.intra_weight_neg = 1.0
        self.intra_weight_pos = 1.0
        self.pos_level = 4
        self.intra_proxy_lr = 0.00001
        self.intra_margin = 0.2
        self.intra_beta_lr = 0.0005
        self.intra_beta_constant = False
        self.intra_beta = 1.2
        self.intra_nu = 0
        self.no_adv_class_reverse = False # False
        self.no_adv_intra_reverse = False  # False
        self.advnet_dim = 512
        self.advnet_decay = 0.000001
        self.use_super = False
        self.not_pretrained = False
        self.not_spliteval = False
        self.k_vals = [1,2,4,8]
        self.arch = 'resnet50'
        self.no_weights = False
        self.source_path = '/export/home/karoth/Datasets'
        self.save_path = os.getcwd()+'/Training_Results'
        self.freq_eval_res = 1
        self.freq_eval_trainvalset = 999
        self.freq_tsne = 999
        self.freq_analyse_inter_intra_dist = 999

        self.adv_detach_target = False
        self.adv_dir = 'res_class' # class_res      res_class
        self.disw = 2000 #500
        self.num_cluster = 200
        self.gpu = 8
        self.savename = 'verify_old_code_280820' # 'cars_200_noise_adv_500_70_90_seed23_origAdvDir_epoch150New'


opt = opt_class()
opt.embed_dim = opt.classembed # + opt.intraclassembed

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

model         = NetworkSuperClass(opt)

opt.input_space, opt.input_range, opt.mean, opt.std = model.input_space, model.input_range, model.mean, model.std
print('{}|{} Setup for {} with {}|{} sampling on {} complete with #weights: {}'.format(opt.class_loss.upper(), opt.intra_loss.upper(), opt.arch.upper(), opt.class_sampling.upper(), \
                                                                                       opt.intra_sampling.upper(),opt.dataset.upper(), aux.gimme_params(model)))

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

if opt.dataset=='online_products' and opt.use_super:
    cluster_labels = losses.random_cluster(opt, super_train_eval_dataloader)
    image_paths    = np.array([x[0] for x in super_train_eval_dataloader.dataset.image_list])
else:
    # cluster_labels = losses.random_cluster(opt, train_eval_dataloader)
    # cluster_labels =np.array([x[1] for x in train_eval_dataloader.dataset.image_list]) # gt labels for label reversal!

    image_paths    = np.array([x[0] for x in train_dataloader.dataset.image_list])
    gt_labels = np.array([x[1] for x in train_dataloader.dataset.image_list])

# clusterdataset     = data.ClusterDataset(image_paths, cluster_labels, opt)
# clusterdataset     = data.RandomizedDataset(image_paths, opt)
clusterdataset     = data.RandomizedDataset_wLabel(image_paths, gt_labels, opt)
# clusterdataset     = data.PartlyRandomizedDataset_wLabel(image_paths, gt_labels, opt, random_prob=0.75)

cluster_dataloader = torch.utils.data.DataLoader(clusterdataset, batch_size=opt.bs, num_workers=opt.kernels, shuffle=True, pin_memory=True, drop_last=True)




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

# eval residual embedding
if opt.freq_eval_res > 0:
    if opt.dataset != 'vehicle_id':
        CSV_log_val_res = aux.CSV_Writer(opt.save_path + '/log_epoch_val_res.csv', ['Epoch', 'NMI', 'Val Recall Sum'] + ['Recall @ {}'.format(k_val) for k_val in  opt.k_vals] + ['Time'])
        CSV_log_trainval_res = aux.CSV_Writer(opt.save_path + '/log_epoch_trainval_res.csv', ['Epoch', 'NMI', 'Val Recall Sum'] + ['Recall @ {}'.format(k_val) for k_val in  opt.k_vals] + ['Time'])


# debug stuff
# CSV_log_triplet_dist = aux.CSV_Writer(opt.save_path +'/log_epoch_sample_triplet_dist.csv', ['Epoch', 'ap_class', 'an_class', 'ap_noise', 'an_noise'])
# CSV_log_rand_triplet_dist = aux.CSV_Writer(opt.save_path +'/log_epoch_rand_triplet_dist.csv', ['Epoch', 'ap_class', 'an_class', 'ap_noise', 'an_noise'])

# CSV_log_class_dist_trainval = aux.CSV_Writer(opt.save_path +'/log_epoch_class_dist_trainval.csv', ['Epoch', 'mean_intra_class', 'mean_inter_class', 'mean_intra_res', 'mean_inter_res'])
# CSV_log_class_dist_val = aux.CSV_Writer(opt.save_path +'/log_epoch_class_dist_val.csv', ['Epoch', 'mean_intra_class', 'mean_inter_class', 'mean_intra_res', 'mean_inter_res'])


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
class_criterion, to_optim = losses.loss_select(opt.class_loss, loss_pars, to_optim)

loss_pars.nu, loss_pars.beta, loss_pars.beta_lr, loss_pars.beta_constant, loss_pars.embed_dim   = opt.intra_nu, opt.intra_beta, opt.intra_beta_lr, opt.intra_beta_constant, opt.embed_dim
loss_pars.margin, loss_pars.loss, loss_pars.sampling, loss_pars.num_classes, loss_pars.proxy_lr = opt.intra_margin, opt.intra_loss, opt.intra_sampling, opt.num_cluster, opt.intra_proxy_lr
loss_pars.weight_pos, loss_pars.weight_neg = opt.intra_weight_pos, opt.intra_weight_neg
loss_pars.pos_level = opt.pos_level
intra_criterion, to_optim = losses.loss_select(opt.intra_loss, loss_pars, to_optim)

if opt.disw:
    adv_criterion = losses.AdvLoss(opt, opt.classembed, opt.intraclassembed, opt.advnet_dim)
    to_optim += [{'params':adv_criterion.parameters(), 'lr':opt.lr, 'weight_decay':opt.advnet_decay}]

_,_ = class_criterion.to(opt.device), intra_criterion.to(opt.device)





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

global best_recall_score_res
best_recall_score_res = 0

global best_recall_score_class
best_recall_score_class = 0


"""============================================================================"""
#################### TRAINER FUNCTION ############################
def train_one_epoch(train_dataloader, cluster_dataloader, model, optimizer, class_criterion, intra_criterion, opt, progress_saver, epoch):
    loss_collect_class, loss_collect_intra, loss_collect_adv = [],[],[]

    # # sampled
    # mean_dist_ap_class = list()
    # mean_dist_an_class = list()
    # mean_dist_ap_res = list()
    # mean_dist_an_res = list()
    #
    # # random
    # rand_mean_dist_ap_class = list()
    # rand_mean_dist_an_class = list()
    # rand_mean_dist_ap_res = list()
    # rand_mean_dist_an_res = list()

    start = time.time()
    data_iterator = tqdm(zip(train_dataloader,cluster_dataloader), desc='Epoch {} Training...'.format(epoch), total=len(train_dataloader))
    for i,((class_labels, input),(cluster_class_labels, cluster_class_labels_gt, cluster_input)) in enumerate(data_iterator):

        #### Train InterClass Embedding
        features  = model(input.to(opt.device))
        loss, sampled_triplets     = class_criterion(features[:,:opt.classembed], class_labels)
        if opt.disw:
            adv_loss = adv_criterion(features[:,:opt.classembed], features[:,opt.classembed:])
            loss = loss+opt.disw*adv_loss
            loss_collect_adv.append(adv_loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_collect_class.append(loss.item())

        # # analyze sampled triplets:
        # features = features.detach().cpu().numpy()
        # triplets = np.asarray([trip for trip in sampled_triplets])
        # # # distances between A,P and A,N
        # dists_ap = np.sqrt(np.sum((features[triplets[:, 0], :opt.classembed] - features[triplets[:, 1], :opt.classembed]) ** 2, axis=1))
        # mean_dist_ap_class.append(np.mean(dists_ap))
        # dists_an = np.sqrt(np.sum((features[triplets[:, 0], :opt.classembed] - features[triplets[:, 2], :opt.classembed]) ** 2, axis=1))
        # mean_dist_an_class.append(np.mean(dists_an))
        # dists_ap = np.sqrt(np.sum((features[triplets[:, 0], opt.classembed:] - features[triplets[:, 1], opt.classembed:]) ** 2, axis=1))
        # mean_dist_ap_res.append(np.mean(dists_ap))
        # dists_an = np.sqrt(np.sum((features[triplets[:, 0], opt.classembed:] - features[triplets[:, 2], opt.classembed:]) ** 2, axis=1))
        # mean_dist_an_res.append(np.mean(dists_an))

        ##### Train IntraClass Embedding
        features  = model(cluster_input.to(opt.device))
        loss, random_triplets      = intra_criterion(features[:,opt.classembed:], cluster_class_labels, cluster_class_labels_gt)
        if opt.disw:
            adv_loss = adv_criterion(features[:,:opt.classembed], features[:,opt.classembed:])
            loss = loss+opt.disw*adv_loss
            loss_collect_adv.append(adv_loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_collect_intra.append(loss.item())

        # # analyze random triplets:
        # features = features.detach().cpu().numpy()
        # triplets = np.asarray([trip for trip in random_triplets])
        # # # distances between A,P and A,N
        # dists_ap = np.sqrt(np.sum((features[triplets[:, 0], :opt.classembed] - features[triplets[:, 1], :opt.classembed])**2, axis=1))
        # rand_mean_dist_ap_class.append(np.mean(dists_ap))
        # dists_an = np.sqrt(np.sum((features[triplets[:, 0], :opt.classembed] - features[triplets[:, 2], :opt.classembed])**2, axis=1))
        # rand_mean_dist_an_class.append(np.mean(dists_an))
        # dists_ap = np.sqrt(np.sum((features[triplets[:, 0], opt.classembed:] - features[triplets[:, 1], opt.classembed:])**2, axis=1))
        # rand_mean_dist_ap_res.append(np.mean(dists_ap))
        # dists_an = np.sqrt(np.sum((features[triplets[:, 0], opt.classembed:] - features[triplets[:, 2], opt.classembed:])**2, axis=1))
        # rand_mean_dist_an_res.append(np.mean(dists_an))


        #####
        if i==len(train_dataloader)-1: data_iterator.set_description('Epoch (Train) {0}: Mean Loss Class [{1:.4f}] | Mean Loss Intra [{2:.4f}] | Mean Adv Loss [{3:.8f}]'.format(
            epoch, np.mean(loss_collect_class), np.mean(loss_collect_intra), np.mean(loss_collect_adv)))

    # update logging
    CSV_log_train.log([epoch, np.mean(loss_collect_class), np.round(time.time()-start,4)])
    progress_saver['Train Loss'].append(np.mean(loss_collect_class))

    ### debug stuff
    # CSV_log_triplet_dist.log([epoch, np.mean(mean_dist_ap_class), np.mean(mean_dist_an_class), np.mean(mean_dist_ap_res), np.mean(mean_dist_an_res)])
    # CSV_log_rand_triplet_dist.log([epoch, np.mean(rand_mean_dist_ap_class), np.mean(rand_mean_dist_an_class), np.mean(rand_mean_dist_ap_res), np.mean(rand_mean_dist_an_res)])



"""============================================================================"""
#################### EVALUATION PROTOCOLS ############################
##### For every dataset besides PKU Vehicle-ID and InShop
def evaluate_standard(val_dataloader, model, optimizer, opt, progress_saver, epoch, data_type='val'):
    global best_recall_score_res
    global best_recall_score_class

    loss_collect = []

    start = time.time()

    with torch.no_grad():
        if opt.dataset!='in-shop':

            # eval class embedding
            NMI, recall_at_ks, feature_coll, image_paths, mean_intra_dist, mean_inter_dist = aux.eval_metrics(model, val_dataloader, opt.device, spliteval=opt.spliteval, epoch=epoch, k_vals = opt.k_vals, opt=opt, embed_type='class')

            # eval residual embedding
            if epoch % opt.freq_eval_res == 0: # evaluate residual embedding
                print('*** Evaluation auxiliary embedding:')
                NMI_res, recall_at_ks_res, feature_coll_res, image_paths_res, mean_intra_dist_res, mean_inter_dist_res = \
                    aux.eval_metrics(model, val_dataloader, opt.device, spliteval=opt.spliteval, epoch=epoch, k_vals=opt.k_vals, opt=opt, embed_type='res')

                # log evaluations of residual embedding
                if data_type == 'val':
                    CSV_log_val_res.log([epoch, NMI_res, np.sum(recall_at_ks_res)] + recall_at_ks_res + [np.round(time.time()-start)])

                    # save checkpoint (ON VAL SET ONLY)
                    if recall_at_ks_res[0] > best_recall_score_res:
                        print('*** !SAVE! Eval (noise, val): {} [top: {}]'.format(round(recall_at_ks_res[0], 4), round(best_recall_score_res, 4)))
                        aux.set_checkpoint(model, opt, epoch, optimizer, opt.save_path, progress_saver, suffix='res')
                        best_recall_score_res = recall_at_ks_res[0]

                elif data_type == 'trainval':
                    CSV_log_trainval_res.log([epoch, NMI_res, np.sum(recall_at_ks_res)] + recall_at_ks_res + [np.round(time.time()-start)])

                # # log distances between classes
                # if epoch % opt.freq_analyse_inter_intra_dist == 0:
                #     if data_type == 'val':
                #         CSV_log_class_dist_val.log([epoch, mean_intra_dist, mean_inter_dist, mean_intra_dist_res, mean_inter_dist_res])
                #     elif data_type == 'trainval':
                #         CSV_log_class_dist_trainval.log([epoch, mean_intra_dist, mean_inter_dist, mean_intra_dist_res, mean_inter_dist_res])


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
def compute_cluster_purity(cluster_ids, gt_labels):
    member_labels = [gt_labels[k] for k in cluster_ids]
    cluster_label = max(set(member_labels), key=member_labels.count)
    purity = member_labels.count(cluster_label) / len(cluster_ids)

    return purity, member_labels, cluster_label


#################### MAIN PART ############################
print('\n-----\n')
for epoch in range(opt.n_epochs):
    print('Model: {}'.format(opt.savename))

    if opt.scheduler=='cosine' and epoch==opt.T_switch:
        opt.T_max    *= opt.T_mul
        opt.T_switch += opt.T_max
        cluster_labels = losses.cluster(opt, train_eval_dataloader, model, mode=opt.mode_main, num_cluster=opt.num_cluster, full=opt.full_embed_cluster, ignore_last=opt.ignore_last)
        cluster_dataloader.dataset.update_labels(cluster_labels)
        scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.T_max, eta_min=5e-8)
    if opt.scheduler!='none': print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_lr())))

    ### Train one epoch
    model.train()
    train_one_epoch(train_dataloader, cluster_dataloader, model, optimizer, class_criterion, intra_criterion, opt, Progress_Saver, epoch)

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
        perplex = [10, 30, 50]
        if epoch % opt.freq_tsne == 0 and epoch > 0:
            print('*** Creating tSNE plots...')
            aux.plot_tSNE(opt, train_eval_dataloader, model, feat='embed', type_data='trainval', n_samples=700, img_size=60,
                          perplex=perplex, img_zoom=0.7, epoch=epoch)
            aux.plot_tSNE(opt, val_dataloader, model, feat='embed', type_data='val', n_samples=700, img_size=60,
                          perplex=perplex, img_zoom=0.7, epoch=epoch)
            print('*** Creating tSNE plots... DONE!')

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
