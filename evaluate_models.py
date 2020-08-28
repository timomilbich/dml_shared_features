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

from sklearn.metrics import pairwise_distances
from sklearn import metrics
import faiss


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
        self.dataset = 'cars196' # cars196   cub200
        self.shared_norm = False
        self.lr = 0.00001
        self.n_epochs = 150
        self.kernels = 8
        self.bs = 112
        self.samples_per_class = 4
        self.seed = 23
        self.scheduler = 'step'
        self.gamma = 0.3
        self.decay = 0.0004
        self.tau = [70, 90] # [70, 90, 120]    [55, 80]
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
        self.intra_loss = 'marginloss_noise'
        self.intra_sampling = 'distance_noise'    #distance_noise
        self.intra_weight_neg = 1.0
        self.intra_weight_pos = 1.0
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
        self.source_path = os.getcwd()+'/Datasets'
        self.save_path = os.getcwd()+'/Analysis/NNs'
        self.freq_eval_res = 1
        self.freq_eval_trainvalset = 25
        self.freq_tsne = 25
        self.freq_analyse_inter_intra_dist = 25

        self.adv_detach_target = False
        self.adv_dir = 'res_class' # class_res      res_class
        self.disw = 500
        self.num_cluster = 200
        self.cluster_update = 1
        self.gpu = 0
        self.model_name = 'cars_200_noise_adv_500_70_90_seed23_origAdvDir_epoch150New'
        # self.model_name = 'cars_200_noise_singleEmbed_70_90_seed23_LabelsOnly'



opt = opt_class()
opt.embed_dim = opt.classembed # +opt.intraclassembed

"""============================================================================"""
opt.source_path += '/'+opt.dataset
opt.save_path   += '/'+opt.dataset
print(' **** use model: {} ****'.format(opt.model_name))

"""============================================================================"""
################### GPU SETTINGS ###########################
os.environ["CUDA_DEVICE_ORDER"]   ="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(opt.gpu)


"""============================================================================"""
##################### NETWORK SETUP ##################
#NOTE: Networks that can be used: 'bninception, resnet50, resnet101, alexnet...'
#>>>>  see import pretrainedmodels; pretrainedmodels.model_names
opt.device    = torch.device('cuda')

model         = NetworkSuperClass(opt)

opt.input_space, opt.input_range, opt.mean, opt.std = model.input_space, model.input_range, model.mean, model.std
print('{}|{} Setup for {} with {}|{} sampling on {} complete with #weights: {}'.format(opt.class_loss.upper(), opt.intra_loss.upper(), opt.arch.upper(), opt.class_sampling.upper(), \
                                                                                       opt.intra_sampling.upper(),opt.dataset.upper(), aux.gimme_params(model)))

_  = model.to(opt.device)


# load weights
weights = torch.load(os.getcwd()+'/Training_Results/{}/{}/checkpoint_res.pth.tar'.format(opt.dataset, opt.model_name))['state_dict']
model.load_state_dict(weights)

# # randomly re-initialize embed
# for idx, module in enumerate(model.modules()):
#     if isinstance(module, nn.Linear):
#         module.weight.data.normal_(0, 0.01)
#         module.bias.data.zero_()

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
# clusterdataset     = data.RandomizedDataset_wLabel(image_paths, gt_labels, opt)


"""============================================================================"""
#################### CHECK MODEL PERFORMANCE ############################
# NMI, recall_at_ks, feature_coll, image_paths, mean_intra_dist, mean_inter_dist = \
#     aux.eval_metrics(model, val_dataloader, opt.device, spliteval=opt.spliteval, epoch=0, k_vals = opt.k_vals, opt=opt, embed_type='res')
#
# print('*** (noise) Eval on validation set: R@1 = {}'.format(recall_at_ks[0]))

"""============================================================================"""
#################### COMPUTE NN ############################

# choose dataset
dataloader = train_eval_dataloader                    # train_eval_dataloader    val_dataloader

feat2use = 'embed'
embed_type = 'res'
opt.spliteval = True

image_paths = np.array([x[0] for x in dataloader.dataset.image_list])
gt_labels = np.array([x[1] for x in dataloader.dataset.image_list])

_ = model.eval()
n_classes = len(dataloader.dataset.avail_classes)

### For all test images, extract features
with torch.no_grad():
    target_labels, feature_coll = [], []
    final_iter = tqdm(dataloader, desc='Computing Evaluation Metrics...')
    image_paths = [x[0] for x in dataloader.dataset.image_list]
    for idx, inp in enumerate(final_iter):
        input_img, target = inp[-1], inp[0]
        target_labels.extend(target.numpy().tolist())

        out = model(input_img.to(opt.device), feat=feat2use)

        if feat2use == 'embed':
            if opt.spliteval:
                if embed_type == 'class':
                    out = out[:, :opt.classembed]
                elif embed_type == 'res':
                    out = out[:, opt.classembed:]
                else:
                    raise Exception('Unknown embed type!')

        feature_coll.extend(out.cpu().detach().numpy().tolist())

    target_labels = np.hstack(target_labels).reshape(-1, 1)
    feature_coll = np.vstack(feature_coll)
    ### TODO CHECK EVAL PART HERE
    # feature_coll  = (feature_coll-np.min(feature_coll))/(np.max(feature_coll)-np.min(feature_coll))
    feature_coll = feature_coll.astype('float32')

    torch.cuda.empty_cache()
    ### Set CPU Cluster index
    cpu_cluster_index = faiss.IndexFlatL2(feature_coll.shape[-1])
    kmeans = faiss.Clustering(feature_coll.shape[-1], n_classes)
    kmeans.niter = 20
    kmeans.min_points_per_centroid = 1
    kmeans.max_points_per_centroid = 1000000000

    ### Train Kmeans
    kmeans.train(feature_coll, cpu_cluster_index)
    computed_centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes, feature_coll.shape[-1])

    ### Assign feature points to clusters
    faiss_search_index = faiss.IndexFlatL2(computed_centroids.shape[-1])
    faiss_search_index.add(computed_centroids)
    _, model_generated_cluster_labels = faiss_search_index.search(feature_coll, 1)

    ### Compute NMI
    NMI = metrics.cluster.normalized_mutual_info_score(model_generated_cluster_labels.reshape(-1),
                                                       target_labels.reshape(-1))

    NMI_adjust = metrics.cluster.adjusted_mutual_info_score(model_generated_cluster_labels.reshape(-1),
                                                       target_labels.reshape(-1))

    ### Recover max(k_vals) nearest neighbours to use for recall computation
    faiss_search_index = faiss.IndexFlatL2(feature_coll.shape[-1])
    faiss_search_index.add(feature_coll)
    ### NOTE: when using the same array for search and base, we need to ignore the first returned element.
    _, k_closest_points = faiss_search_index.search(feature_coll, int(np.max(opt.k_vals) + 1))
    k_closest_classes = target_labels.reshape(-1)[k_closest_points[:, 1:]]

    ### Compute Recall
    recall_all_k = []
    for k in opt.k_vals:
        recall_at_k = np.sum([1 for target, recalled_predictions in zip(target_labels, k_closest_classes) if
                              target in recalled_predictions[:k]]) / len(target_labels)
        recall_all_k.append(recall_at_k)
    recall_str = ', '.join('@{0}: {1:.4f}'.format(k, rec) for k, rec in zip(opt.k_vals, recall_all_k))

    message = 'Epoch (Test) {0}: NMI [{1:.4f}] | NMI (adjust) [{2:.4f}] | Recall [{3}]'.format('Eval', NMI, NMI_adjust, recall_str)
    print(message)


