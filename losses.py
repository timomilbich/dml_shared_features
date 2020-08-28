import warnings
warnings.filterwarnings("ignore")

import torch, random, itertools as it, numpy as np, faiss, random
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from sklearn.preprocessing import normalize
from PIL import Image
import time as time
import os
import copy




"""================================================================================================="""
def loss_select(loss, opt, to_optim):
    if loss=='triplet':
        loss_params  = {'margin':opt.margin}
        criterion    = TripletLoss(**loss_params)

    elif loss=='marginloss':
        loss_params  = {'margin':opt.margin, 'nu': opt.nu, 'beta':opt.beta, 'n_classes':opt.num_classes,
                        'beta_constant': opt.beta_constant, 'sampling_method': opt.sampling}
        criterion    = MarginLoss(**loss_params)
        to_optim    += [{'params':criterion.parameters(), 'lr':opt.beta_lr, 'weight_decay':0}]

    elif loss=='proxynca':
        loss_params  = {'num_proxies':opt.num_classes, 'embedding_dim':opt.classembed if 'num_cluster' in vars(opt).keys() else opt.embed_dim}
        criterion    = ProxyNCALoss(**loss_params)
        to_optim    += [{'params':criterion.parameters(), 'lr':opt.proxy_lr}]

    elif loss == 'marginloss_debug':
        loss_params = {'margin': opt.margin, 'nu': opt.nu, 'beta': opt.beta, 'n_classes': opt.num_classes,
                       'beta_constant': opt.beta_constant}
        criterion = MarginLossDEBUG(**loss_params)
        to_optim += [{'params': criterion.parameters(), 'lr': opt.beta_lr, 'weight_decay': 0}]

    elif loss == 'marginloss_noise':
        loss_params = {'margin': opt.margin, 'nu': opt.nu, 'beta': opt.beta, 'n_classes': opt.num_classes,
                       'beta_constant': opt.beta_constant, 'weight_pos': opt.weight_pos, 'weight_neg': opt.weight_neg, 'sampling_method': opt.sampling}
        criterion = MarginLoss_noise(**loss_params)
        to_optim += [{'params': criterion.parameters(), 'lr': opt.beta_lr, 'weight_decay': 0}]

    elif loss == 'marginloss_noise_gor':
        loss_params = {'margin': opt.margin, 'nu': opt.nu, 'beta': opt.beta, 'n_classes': opt.num_classes, 'alpha_gor': opt.alpha_gor,
                       'beta_constant': opt.beta_constant, 'weight_pos': opt.weight_pos, 'weight_neg': opt.weight_neg, 'sampling_method': opt.sampling}
        criterion = MarginLoss_noise_gor(**loss_params)
        to_optim += [{'params': criterion.parameters(), 'lr': opt.beta_lr, 'weight_decay': 0}]

    elif loss == 'marginloss_noise_constrainPos':
        loss_params = {'margin': opt.margin, 'nu': opt.nu, 'beta': opt.beta, 'n_classes': opt.num_classes, 'pos_level': opt.pos_level,
                       'beta_constant': opt.beta_constant, 'weight_pos': opt.weight_pos, 'weight_neg': opt.weight_neg, 'sampling_method': opt.sampling}
        criterion = MarginLoss_noise_constrainPos(**loss_params)
        to_optim += [{'params': criterion.parameters(), 'lr': opt.beta_lr, 'weight_decay': 0}]

    else:
        raise Exception('Loss {} not available!'.format(opt.loss))

    return criterion, to_optim


"""================================================================================================="""
### Sampler() holds all possible triplet sampling options: random, SemiHardNegative & Distance-Weighted.
class Sampler():
    def __init__(self, method='random'):
        self.method = method
        if method=='semihard':
            self.give = self.semihardsampling
        elif method=='distance':
            self.give = self.distanceweightedsampling
        elif method=='random':
            self.give = self.randomsampling
        elif method=='distance_debug':
            self.give = self.distanceweightedsampling_DEBUG
        elif method == 'random_label_reversal':
            self.give = self.random_label_reversal
        elif method == 'distance_label_reversal':
            self.give = self.distance_label_reversal
        elif method == 'random_class_balanced':
            self.give = self.random_class_balanced
        elif method=='distance_noise':
            self.give = self.distanceweightedsampling_noise
        elif method == 'distance_noise_pullNeg':
            self.give = self.distanceweightedsampling_noise_pullNeg
        elif method == 'distance_noise_pullPos':
            self.give = self.distanceweightedsampling_noise_pullPos
        elif method == 'distance_noise_constrainPos':
            self.give = self.distanceweightedsampling_noise_constrainPos

    def randomsampling(self, batch, labels):
        """
        This methods finds all available triplets in a batch given by the classes provided in labels, and randomly
        selects batch.batchsize triplets.
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach().numpy()
        unique_classes = np.unique(labels)
        indices        = np.arange(len(batch))
        class_dict     = {i:indices[labels==i] for i in unique_classes}

        sampled_triplets = [list(it.product([x],[x],[y for y in unique_classes if x!=y])) for x in unique_classes]
        sampled_triplets = [x for y in sampled_triplets for x in y]

        sampled_triplets = [[x for x in list(it.product(*[class_dict[j] for j in i])) if x[0]!=x[1]] for i in sampled_triplets]
        sampled_triplets = [x for y in sampled_triplets for x in y]

        #NOTE: The number of possible triplets is given by #unique_classes*(2*(samples_per_class-1)!)*(#unique_classes-1)*samples_per_class
        sampled_triplets = random.sample(sampled_triplets, batch.shape[0])
        return sampled_triplets

    def semihardsampling(self, batch, labels):
        if isinstance(labels, torch.Tensor): labels = labels.detach().numpy()
        bs = batch.size(0)
        #Return distance matrix for all elements in batch (BSxBS)
        distances = squareform(pdist(batch.detach().cpu().numpy()))

        positives, negatives = [], []
        anchors = []
        for i in range(bs):
            l, d = labels[i], distances[i]
            #1 for batchelements with label l
            neg = labels!=l; pos = labels==l
            #0 for current anchor
            pos[i] = False
            #Find negatives that violate triplet constraint semi-negatives
            neg_mask = np.logical_and(neg,d<d[pos].max())
            #Find positives that violate triplet constraint semi-hardly
            pos_mask = np.logical_and(pos,d>d[neg].min())

            if pos_mask.sum()>0:
                positives.append(np.random.choice(np.where(pos_mask)[0]))
            else:
                positives.append(np.random.choice(np.where(pos)[0]))

            if neg_mask.sum()>0:
                negatives.append(np.random.choice(np.where(neg_mask)[0]))
            else:
                negatives.append(np.random.choice(np.where(neg)[0]))

        sampled_triplets = [(anchor, positive, negative) for anchor, positive, negative in zip(anchors, positives, negatives)]
        return sampled_triplets

    def pdist(self, A, eps = 1e-4):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min = 0)
        return res.clamp(min = eps).sqrt()

    def distanceweightedsampling(self, batch, labels, lower_cutoff=0.5, upper_cutoff=1.4, orig_labels=None, compute_dist=False):
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        bs = batch.shape[0]

        distances    = self.pdist(batch.detach()).clamp(min=lower_cutoff)

        def inverse_sphere_distances(batch, dist, labels, anchor_label, orig_labels=None, anchor_orig_label=None):
            bs,dim       = len(dist),batch.shape[-1]

            #negated log-distribution of distances of unit sphere in dimension <dim>
            log_q_d_inv = ((2.0 - float(dim)) * torch.log(dist) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (dist.pow(2))))
            log_q_d_inv[np.where(labels==anchor_label)[0]] = 0

            q_d_inv     = torch.exp(log_q_d_inv - torch.max(log_q_d_inv)) # - max(log) for stability
            q_d_inv[np.where(labels==anchor_label)[0]] = 0

            ### NOTE: Cutting of values with high distances made the results slightly worse.
            # q_d_inv[np.where(dist>upper_cutoff)[0]]    = 0

            q_d_inv = q_d_inv/q_d_inv.sum()
            return q_d_inv.detach().cpu().numpy()

        positives, negatives = [],[]
        labels_visited = []
        anchors = []

        for i in range(bs):
            neg = labels!=labels[i]; pos = labels==labels[i]
            # q_d_inv = inverse_sphere_distances(batch, distances[i], labels, labels[i], orig_labels, orig_labels[i])
            q_d_inv = inverse_sphere_distances(batch, distances[i], labels, labels[i])
            # for pos_ix in np.where(pos)[0]:
            #     if pos_ix!=i:
            #         anchors.append(i)
            #         positives.append(pos_ix)
            #         #Sample negatives by distance
            #         negatives.append(np.random.choice(bs,p=q_d_inv))
            #Sample positives randomly
            pos[i] = 0
            positives.append(np.random.choice(np.where(pos)[0]))

            #Sample negatives by distance
            negatives.append(np.random.choice(bs,p=q_d_inv))

        # sampled_triplets = [(a,p,n) for a,p,n in zip(anchors, positives, negatives)]
        sampled_triplets = [(a,p,n) for a,p,n in zip(list(range(bs)), positives, negatives)]
        return sampled_triplets

    def distanceweightedsampling_noise(self, batch, labels, gt_labels, lower_cutoff=0.5, upper_cutoff=1.4, orig_labels=None, compute_dist=False):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            gt_labels = gt_labels.detach().cpu().numpy()

        bs = batch.shape[0]

        distances    = self.pdist(batch.detach()).clamp(min=lower_cutoff)

        def inverse_sphere_distances(batch, dist, labels, anchor_label, gt_labels, anchor_gt_label, orig_labels=None, anchor_orig_label=None):
            bs,dim       = len(dist),batch.shape[-1]

            #negated log-distribution of distances of unit sphere in dimension <dim>
            log_q_d_inv = ((2.0 - float(dim)) * torch.log(dist) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (dist.pow(2))))
            log_q_d_inv[np.where(labels==anchor_label)[0]] = 0

            q_d_inv     = torch.exp(log_q_d_inv - torch.max(log_q_d_inv)) # - max(log) for stability

            # set sampling prob of surrogate positives to zero
            q_d_inv[np.where(labels==anchor_label)[0]] = 0

            # set sampling prob of gt positives to zero
            q_d_inv[np.where(gt_labels==anchor_gt_label)[0]] = 0

            ### NOTE: Cutting of values with high distances made the results slightly worse.
            # q_d_inv[np.where(dist>upper_cutoff)[0]]    = 0

            q_d_inv = q_d_inv/q_d_inv.sum()
            return q_d_inv.detach().cpu().numpy()

        positives, negatives = [],[]
        for i in range(bs):
            neg = labels!=labels[i];
            pos = labels==labels[i]
            q_d_inv = inverse_sphere_distances(batch, distances[i], labels, labels[i], gt_labels, gt_labels[i])

            #Sample positives randomly
            pos[i] = 0
            positives.append(np.random.choice(np.where(pos)[0]))

            #Sample negatives by distance
            negatives.append(np.random.choice(bs, p=q_d_inv))

        # sampled_triplets = [(a,p,n) for a,p,n in zip(anchors, positives, negatives)]
        sampled_triplets = [(a,p,n) for a,p,n in zip(list(range(bs)), positives, negatives)]
        return sampled_triplets

    def distanceweightedsampling_noise_constrainPos(self, batch, labels, gt_labels, lower_cutoff=0.5, pos_level=1, upper_cutoff=1.4, orig_labels=None, compute_dist=False):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            gt_labels = gt_labels.detach().cpu().numpy()

        bs = batch.shape[0]

        distances    = self.pdist(batch.detach()).clamp(min=lower_cutoff)

        def inverse_sphere_distances(batch, dist, labels, anchor_label, gt_labels, anchor_gt_label, orig_labels=None, anchor_orig_label=None):
            bs,dim       = len(dist),batch.shape[-1]

            #negated log-distribution of distances of unit sphere in dimension <dim>
            log_q_d_inv = ((2.0 - float(dim)) * torch.log(dist) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (dist.pow(2))))
            log_q_d_inv[np.where(labels==anchor_label)[0]] = 0

            q_d_inv     = torch.exp(log_q_d_inv - torch.max(log_q_d_inv)) # - max(log) for stability

            # set sampling prob of surrogate positives to zero
            q_d_inv[np.where(labels==anchor_label)[0]] = 0

            # set sampling prob of gt positives to zero
            q_d_inv[np.where(gt_labels==anchor_gt_label)[0]] = 0

            ### NOTE: Cutting of values with high distances made the results slightly worse.
            # q_d_inv[np.where(dist>upper_cutoff)[0]]    = 0

            q_d_inv = q_d_inv/q_d_inv.sum()
            return q_d_inv.detach().cpu().numpy()

        positives, negatives = [],[]
        for i in range(bs):
            neg = labels!=labels[i];
            pos = labels==labels[i]
            q_d_inv = inverse_sphere_distances(batch, distances[i], labels, labels[i], gt_labels, gt_labels[i])

            #Sample positives randomly based on pos_level
            pos[i] = 0
            ids_pos = np.where(pos)[0]
            dists_pos = distances[i, ids_pos].cpu().detach().numpy()
            pos_cands = ids_pos[np.argsort(dists_pos)[0:pos_level]]
            positives.append(np.random.choice(pos_cands))

            #Sample negatives by distance
            negatives.append(np.random.choice(bs, p=q_d_inv))

        # sampled_triplets = [(a,p,n) for a,p,n in zip(anchors, positives, negatives)]
        sampled_triplets = [(a,p,n) for a,p,n in zip(list(range(bs)), positives, negatives)]
        return sampled_triplets

    def distanceweightedsampling_noise_pullNeg(self, batch, labels, gt_labels, lower_cutoff=0.5, upper_cutoff=1.4, orig_labels=None, compute_dist=False):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            gt_labels = gt_labels.detach().cpu().numpy()

        bs = batch.shape[0]

        distances    = self.pdist(batch.detach()).clamp(min=lower_cutoff)

        def inverse_sphere_distances(batch, dist, labels, anchor_label, gt_labels, anchor_gt_label, pos_select, orig_labels=None, anchor_orig_label=None):
            bs,dim       = len(dist),batch.shape[-1]

            #negated log-distribution of distances of unit sphere in dimension <dim>
            log_q_d_inv = ((2.0 - float(dim)) * torch.log(dist) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (dist.pow(2))))
            log_q_d_inv[np.where(labels==anchor_label)[0]] = 0

            q_d_inv     = torch.exp(log_q_d_inv - torch.max(log_q_d_inv)) # - max(log) for stability

            # set sampling prob of surrogate positives to zero
            q_d_inv[np.where(labels==anchor_label)[0]] = 0

            # set sampling prob of gt positives to zero
            q_d_inv[np.where(gt_labels==anchor_gt_label)[0]] = 0

            # consider only negatives which are closer than the selected positive
            q_d_inv_constrained = copy.deepcopy(q_d_inv)
            min_neg_dist = dist[pos_select].detach()
            q_d_inv_constrained[dist >= min_neg_dist] = 0.0
            q_d_inv_constrained = q_d_inv_constrained/ q_d_inv_constrained.sum()

            if not torch.isnan(q_d_inv_constrained[0]):
                return q_d_inv_constrained.detach().cpu().numpy()
            else: # if no negatives are left because of the constraint => nans => use the original probabilitiy distribution
                q_d_inv = q_d_inv/ q_d_inv.sum()
                return q_d_inv.detach().cpu().numpy()

        positives, negatives = [],[]
        for i in range(bs):
            neg = labels!=labels[i];
            pos = labels==labels[i]

            #Sample positives randomly
            pos[i] = 0
            pos_select = np.random.choice(np.where(pos)[0])
            positives.append(pos_select)

            q_d_inv = inverse_sphere_distances(batch, distances[i], labels, labels[i], gt_labels, gt_labels[i], pos_select)

            #Sample negatives by distance
            negatives.append(np.random.choice(bs, p=q_d_inv))

        # sampled_triplets = [(a,p,n) for a,p,n in zip(anchors, positives, negatives)]
        sampled_triplets = [(a,p,n) for a,p,n in zip(list(range(bs)), positives, negatives)]
        return sampled_triplets

    def distanceweightedsampling_noise_pullPos(self, batch, labels, gt_labels, lower_cutoff=0.5, upper_cutoff=1.4, orig_labels=None, compute_dist=False):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            gt_labels = gt_labels.detach().cpu().numpy()

        bs = batch.shape[0]

        distances    = self.pdist(batch.detach()).clamp(min=lower_cutoff)

        def inverse_sphere_distances(batch, dist, labels, anchor_label, gt_labels, anchor_gt_label, pos_select, orig_labels=None, anchor_orig_label=None):
            bs,dim       = len(dist),batch.shape[-1]

            #negated log-distribution of distances of unit sphere in dimension <dim>
            log_q_d_inv = ((2.0 - float(dim)) * torch.log(dist) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (dist.pow(2))))
            log_q_d_inv[np.where(labels==anchor_label)[0]] = 0

            q_d_inv     = torch.exp(log_q_d_inv - torch.max(log_q_d_inv)) # - max(log) for stability

            # set sampling prob of surrogate positives to zero
            q_d_inv[np.where(labels==anchor_label)[0]] = 0

            # set sampling prob of gt positives to zero
            q_d_inv[np.where(gt_labels==anchor_gt_label)[0]] = 0

            # consider only negatives which are more far than the selected positive
            q_d_inv_constrained = copy.deepcopy(q_d_inv)
            max_neg_dist = dist[pos_select].detach()
            q_d_inv_constrained[dist <= max_neg_dist] = 0.0
            q_d_inv_constrained = q_d_inv_constrained/ q_d_inv_constrained.sum()

            if not torch.isnan(q_d_inv_constrained[0]):
                return q_d_inv_constrained.detach().cpu().numpy()
            else: # if no negatives are left because of the constraint => nans => use the original probabilitiy distribution
                q_d_inv = q_d_inv/ q_d_inv.sum()
                return q_d_inv.detach().cpu().numpy()

        positives, negatives = [],[]
        for i in range(bs):
            neg = labels!=labels[i];
            pos = labels==labels[i]

            #Sample positives randomly
            pos[i] = 0
            pos_select = np.random.choice(np.where(pos)[0])
            positives.append(pos_select)

            q_d_inv = inverse_sphere_distances(batch, distances[i], labels, labels[i], gt_labels, gt_labels[i], pos_select)

            #Sample negatives by distance
            negatives.append(np.random.choice(bs, p=q_d_inv))

        # sampled_triplets = [(a,p,n) for a,p,n in zip(anchors, positives, negatives)]
        sampled_triplets = [(a,p,n) for a,p,n in zip(list(range(bs)), positives, negatives)]
        return sampled_triplets

    def distanceweightedsampling_DEBUG(self, batch, labels, gt_labels, lower_cutoff=0.5, upper_cutoff=1.4, orig_labels=None, compute_dist=False):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            gt_labels = gt_labels.detach().cpu().numpy()

        bs = batch.shape[0]

        distances    = self.pdist(batch.detach()).clamp(min=lower_cutoff)

        def inverse_sphere_distances(batch, dist, labels, anchor_label, orig_labels=None, anchor_orig_label=None):
            bs,dim       = len(dist),batch.shape[-1]

            #negated log-distribution of distances of unit sphere in dimension <dim>
            log_q_d_inv = ((2.0 - float(dim)) * torch.log(dist) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (dist.pow(2))))
            log_q_d_inv[np.where(labels==anchor_label)[0]] = 0

            q_d_inv     = torch.exp(log_q_d_inv - torch.max(log_q_d_inv)) # - max(log) for stability
            q_d_inv[np.where(labels==anchor_label)[0]] = 0

            ### NOTE: Cutting of values with high distances made the results slightly worse.
            # q_d_inv[np.where(dist>upper_cutoff)[0]]    = 0

            q_d_inv = q_d_inv/q_d_inv.sum()
            return q_d_inv.detach().cpu().numpy()

        positives, negatives = [],[]
        labels_visited = []
        anchors = []

        for i in range(bs):
            neg = labels!=labels[i]; pos = labels==labels[i]
            # q_d_inv = inverse_sphere_distances(batch, distances[i], labels, labels[i], orig_labels, orig_labels[i])
            q_d_inv = inverse_sphere_distances(batch, distances[i], labels, labels[i])
            # for pos_ix in np.where(pos)[0]:
            #     if pos_ix!=i:
            #         anchors.append(i)
            #         positives.append(pos_ix)
            #         #Sample negatives by distance
            #         negatives.append(np.random.choice(bs,p=q_d_inv))

            #Sample positives randomly
            pos[i] = 0
            positives.append(np.random.choice(np.where(pos)[0]))

            #Sample negatives by distance
            negatives.append(np.random.choice(bs,p=q_d_inv))

        # sampled_triplets = [(a,p,n) for a,p,n in zip(anchors, positives, negatives)]
        sampled_triplets = [(a,p,n) for a,p,n in zip(list(range(bs)), positives, negatives)]
        triplet_labels = [(gt_labels[a], gt_labels[p], gt_labels[n]) for a,p,n in zip(list(range(bs)), positives, negatives) ]

        return sampled_triplets, triplet_labels

    def random_label_reversal(self, batch, labels):
        """
        This methods finds all available triplets in a batch given by the classes provided in labels, and randomly
        selects batch.batchsize triplets.
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach().numpy()
        unique_classes = np.unique(labels)
        indices = np.arange(len(batch))
        class_dict = {i: indices[labels == i] for i in unique_classes}

        sampled_triplets = [list(it.product([x], [x], [y for y in unique_classes if x != y])) for x in unique_classes]
        sampled_triplets = [x for y in sampled_triplets for x in y]

        sampled_triplets = [[x for x in list(it.product(*[class_dict[j] for j in i])) if x[0] != x[1]] for i in
                            sampled_triplets]
        sampled_triplets = [x for y in sampled_triplets for x in y]

        # NOTE: The number of possible triplets is given by #unique_classes*(2*(samples_per_class-1)!)*(#unique_classes-1)*samples_per_class
        sampled_triplets = random.sample(sampled_triplets, batch.shape[0])

        # change positive and negative
        sampled_triplets_rev = [(trip[0], trip[2], trip[1]) for trip in sampled_triplets]

        return sampled_triplets_rev

    def distance_label_reversal(self, batch, labels, lower_cutoff=0.5, upper_cutoff=1.4, orig_labels=None, compute_dist=False):
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        bs = batch.shape[0]

        distances    = self.pdist(batch.detach()).clamp(min=lower_cutoff)

        def inverse_sphere_distances(batch, dist, labels, anchor_label, orig_labels=None, anchor_orig_label=None):
            bs,dim       = len(dist),batch.shape[-1]

            #negated log-distribution of distances of unit sphere in dimension <dim>
            log_q_d_inv = ((2.0 - float(dim)) * torch.log(dist) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (dist.pow(2))))
            log_q_d_inv[np.where(labels==anchor_label)[0]] = 0

            q_d_inv     = torch.exp(log_q_d_inv - torch.max(log_q_d_inv)) # - max(log) for stability
            q_d_inv[np.where(labels==anchor_label)[0]] = 0

            ### NOTE: Cutting of values with high distances made the results slightly worse.
            # q_d_inv[np.where(dist>upper_cutoff)[0]]    = 0

            q_d_inv = q_d_inv/q_d_inv.sum()
            return q_d_inv.detach().cpu().numpy()

        positives, negatives = [],[]
        labels_visited = []
        anchors = []

        for i in range(bs):
            neg = labels!=labels[i]; pos = labels==labels[i]
            # q_d_inv = inverse_sphere_distances(batch, distances[i], labels, labels[i], orig_labels, orig_labels[i])
            q_d_inv = inverse_sphere_distances(batch, distances[i], labels, labels[i])
            # for pos_ix in np.where(pos)[0]:
            #     if pos_ix!=i:
            #         anchors.append(i)
            #         positives.append(pos_ix)
            #         #Sample negatives by distance
            #         negatives.append(np.random.choice(bs,p=q_d_inv))
            #Sample positives randomly
            pos[i] = 0
            positives.append(np.random.choice(np.where(pos)[0]))

            #Sample negatives by distance
            negatives.append(np.random.choice(bs,p=q_d_inv))

        # sampled_triplets = [(a,p,n) for a,p,n in zip(anchors, positives, negatives)]
        sampled_triplets = [(a,p,n) for a,p,n in zip(list(range(bs)), positives, negatives)]

        # change positive and negative
        sampled_triplets_rev = [(trip[0], trip[2], trip[1]) for trip in sampled_triplets]

        return sampled_triplets_rev

    def random_class_balanced(self, batch, labels):
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        bs = batch.shape[0]
        positives, negatives = [],[]

        for i in range(bs):
            neg = labels!=labels[i]; pos = labels==labels[i]
            pos[i] = 0
            # sample random positives
            positives.append(np.random.choice(np.where(pos)[0]))

            #sample random negatives
            negatives.append(np.random.choice(np.where(neg)[0]))

        sampled_triplets = [(a,p,n) for a,p,n in zip(list(range(bs)), positives, negatives)]
        return sampled_triplets



"""================================================================================================="""
### Standard Triplet Loss, finds triplets in Mini-batches.
class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1, sampling_method='random', size_average=False):
        """
        Args:
            margin:             Triplet Margin.
            triplets_per_batch: A batch allows for multitudes of triplets to use. This gives the number
                                if triplets to sample from.
        """
        super(TripletLoss, self).__init__()
        self.margin             = margin
        self.size_average       = size_average
        self.sampler            = Sampler(method=sampling_method)

    def triplet_distance(self, anchor, positive, negative):
        return torch.nn.functional.relu((anchor-positive).pow(2).sum()-(anchor-negative).pow(2).sum()+self.margin)

    def forward(self, batch, labels, sampled_triplets=None):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels:  nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
            sampled_triplets: Optional: Provided pre-sampled triplets
        """
        if sampled_triplets is None: sampled_triplets = self.sampler.give(batch, labels)
        loss             = torch.stack([self.triplet_distance(batch[triplet[0],:],batch[triplet[1],:],batch[triplet[2],:]) for triplet in sampled_triplets])

        if self.size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)




"""================================================================================================="""
### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well. DEBUG!!!
class MarginLossDEBUG(torch.nn.Module):
    def __init__(self, margin=0.2, nu=0, beta=1.2, n_classes=100, beta_constant=False, sampling_method='distance_debug'):
        """
        Args:
            margin:             Triplet Margin.
            nu:                 Regularisation Parameter for beta values if they are learned.
            beta:               Class-Margin values.
            n_classes:          Number of different classes during training.
        """
        super(MarginLossDEBUG, self).__init__()
        self.margin             = margin
        self.n_classes          = n_classes
        self.beta_constant     = beta_constant

        self.beta = beta if beta_constant else torch.nn.Parameter(torch.ones(n_classes)*beta)

        self.nu                 = nu

        self.sampler            = Sampler(method=sampling_method)


    def forward(self, batch, labels, gt_labels, sampled_triplets=None):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            gt_labels = gt_labels.detach().cpu().numpy()

        ###Setup for (learnable) class margin beta.
        if self.nu: beta_regularisation_loss = self.nu*torch.sum(self.beta)

        if sampled_triplets is None: sampled_triplets, triplet_labels = self.sampler.give(batch, labels, gt_labels)

        d_ap = torch.stack([((batch[triplet[0],:]-batch[triplet[1],:]).pow(2).sum()+1e-8).pow(1/2) for triplet in sampled_triplets])
        d_an = torch.stack([((batch[triplet[0],:]-batch[triplet[2],:]).pow(2).sum()+1e-8).pow(1/2) for triplet in sampled_triplets])

        if self.beta_constant:
            beta = self.beta
        else:
            beta = torch.stack([self.beta[labels[triplet[0]]] for triplet in sampled_triplets]).type(torch.cuda.FloatTensor)

        pos_loss = torch.nn.functional.relu(d_ap-beta+self.margin)
        neg_loss = torch.nn.functional.relu(beta-d_an+self.margin)

        ###TODO: Adjust to cuda float
        # pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).type(torch.FloatTensor)
        pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).type(torch.cuda.FloatTensor)

        if pair_count == 0.:
            loss = torch.sum(pos_loss+neg_loss)
        else:
            loss = torch.sum(pos_loss+neg_loss)/pair_count

        if self.nu: loss = loss + beta_regularisation_loss.type(torch.cuda.FloatTensor)

        return loss, triplet_labels



"""================================================================================================="""
### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class MarginLoss(torch.nn.Module):
    def __init__(self, margin=0.2, nu=0, beta=1.2, n_classes=100, beta_constant=False, sampling_method='distance'):
        """
        Args:
            margin:             Triplet Margin.
            nu:                 Regularisation Parameter for beta values if they are learned.
            beta:               Class-Margin values.
            n_classes:          Number of different classes during training.
        """
        super(MarginLoss, self).__init__()
        self.margin             = margin
        self.n_classes          = n_classes
        self.beta_constant     = beta_constant

        self.beta = beta if beta_constant else torch.nn.Parameter(torch.ones(n_classes)*beta)

        self.nu                 = nu

        self.sampler            = Sampler(method=sampling_method)


    def forward(self, batch, labels, sampled_triplets=None):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()

        ###Setup for (learnable) class margin beta.
        if self.nu: beta_regularisation_loss = self.nu*torch.sum(self.beta)

        if sampled_triplets is None: sampled_triplets = self.sampler.give(batch, labels)

        d_ap = torch.stack([((batch[triplet[0],:]-batch[triplet[1],:]).pow(2).sum()+1e-8).pow(1/2) for triplet in sampled_triplets])
        d_an = torch.stack([((batch[triplet[0],:]-batch[triplet[2],:]).pow(2).sum()+1e-8).pow(1/2) for triplet in sampled_triplets])

        if self.beta_constant:
            beta = self.beta
        else:
            beta = torch.stack([self.beta[labels[triplet[0]]] for triplet in sampled_triplets]).type(torch.cuda.FloatTensor)

        pos_loss = torch.nn.functional.relu(d_ap-beta+self.margin)
        neg_loss = torch.nn.functional.relu(beta-d_an+self.margin)

        ###TODO: Adjust to cuda float
        # pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).type(torch.FloatTensor)
        pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).type(torch.cuda.FloatTensor)

        if pair_count == 0.:
            loss = torch.sum(pos_loss+neg_loss)
        else:
            loss = torch.sum(pos_loss+neg_loss)/pair_count

        if self.nu: loss = loss + beta_regularisation_loss.type(torch.cuda.FloatTensor)

        return loss, sampled_triplets


"""================================================================================================="""
### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class MarginLoss_noise(torch.nn.Module):
    def __init__(self, margin=0.2, nu=0, beta=1.2, n_classes=100, beta_constant=False, sampling_method='distance_noise', weight_neg=1.0, weight_pos=1.0):
        """
        Args:
            margin:             Triplet Margin.
            nu:                 Regularisation Parameter for beta values if they are learned.
            beta:               Class-Margin values.
            n_classes:          Number of different classes during training.
        """
        super(MarginLoss_noise, self).__init__()
        self.margin             = margin
        self.n_classes          = n_classes
        self.beta_constant     = beta_constant
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg

        self.beta = beta if beta_constant else torch.nn.Parameter(torch.ones(n_classes)*beta)

        self.nu                 = nu

        self.sampler            = Sampler(method=sampling_method)


    def forward(self, batch, labels, gt_labels, sampled_triplets=None):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            gt_labels = gt_labels.detach().cpu().numpy()

        ###Setup for (learnable) class margin beta.
        if self.nu: beta_regularisation_loss = self.nu*torch.sum(self.beta)

        if sampled_triplets is None:
            sampled_triplets = self.sampler.give(batch, labels, gt_labels)

        d_ap = torch.stack([((batch[triplet[0],:]-batch[triplet[1],:]).pow(2).sum()+1e-8).pow(1/2) for triplet in sampled_triplets])
        d_an = torch.stack([((batch[triplet[0],:]-batch[triplet[2],:]).pow(2).sum()+1e-8).pow(1/2) for triplet in sampled_triplets])

        if self.beta_constant:
            beta = self.beta
        else:
            beta = torch.stack([self.beta[labels[triplet[0]]] for triplet in sampled_triplets]).type(torch.cuda.FloatTensor)

        pos_loss = torch.nn.functional.relu(d_ap-beta+self.margin)
        neg_loss = torch.nn.functional.relu(beta-d_an+self.margin)

        ###TODO: Adjust to cuda float
        # pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).type(torch.FloatTensor)
        pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).type(torch.cuda.FloatTensor)

        # if pair_count == 0.:
        #     loss = torch.sum(pos_loss+neg_loss)
        # else:
        #     loss = torch.sum(pos_loss+neg_loss)/pair_count

        if pair_count == 0.:
            loss = torch.sum(self.weight_pos * pos_loss + self.weight_neg * neg_loss)
        else:
            loss = torch.sum(self.weight_pos * pos_loss + self.weight_neg * neg_loss) / pair_count

        if self.nu: loss = loss + beta_regularisation_loss.type(torch.cuda.FloatTensor)

        return loss, sampled_triplets


"""================================================================================================="""
### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class MarginLoss_noise_gor(torch.nn.Module):
    def __init__(self, margin=0.2, nu=0, beta=1.2, n_classes=100, beta_constant=False, sampling_method='distance_noise', weight_neg=1.0, weight_pos=1.0, alpha_gor=1.0):
        """
        Args:
            margin:             Triplet Margin.
            nu:                 Regularisation Parameter for beta values if they are learned.
            beta:               Class-Margin values.
            n_classes:          Number of different classes during training.
        """
        super(MarginLoss_noise_gor, self).__init__()
        self.margin             = margin
        self.n_classes          = n_classes
        self.beta_constant     = beta_constant
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg
        self.alpha_gor = alpha_gor

        self.beta = beta if beta_constant else torch.nn.Parameter(torch.ones(n_classes)*beta)

        self.nu                 = nu

        self.sampler            = Sampler(method=sampling_method)


    def forward(self, batch, labels, gt_labels, sampled_triplets=None):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            gt_labels = gt_labels.detach().cpu().numpy()

        ###Setup for (learnable) class margin beta.
        if self.nu: beta_regularisation_loss = self.nu*torch.sum(self.beta)

        if sampled_triplets is None:
            sampled_triplets = self.sampler.give(batch, labels, gt_labels)

        d_ap = torch.stack([((batch[triplet[0],:]-batch[triplet[1],:]).pow(2).sum()+1e-8).pow(1/2) for triplet in sampled_triplets])
        d_an = torch.stack([((batch[triplet[0],:]-batch[triplet[2],:]).pow(2).sum()+1e-8).pow(1/2) for triplet in sampled_triplets])

        if self.beta_constant:
            beta = self.beta
        else:
            beta = torch.stack([self.beta[labels[triplet[0]]] for triplet in sampled_triplets]).type(torch.cuda.FloatTensor)

        pos_loss = torch.nn.functional.relu(d_ap-beta+self.margin)
        neg_loss = torch.nn.functional.relu(beta-d_an+self.margin)

        ###TODO: Adjust to cuda float
        # pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).type(torch.FloatTensor)
        pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).type(torch.cuda.FloatTensor)

        # if pair_count == 0.:
        #     loss = torch.sum(pos_loss+neg_loss)
        # else:
        #     loss = torch.sum(pos_loss+neg_loss)/pair_count

        if pair_count == 0.:
            loss = torch.sum(self.weight_pos * pos_loss + self.weight_neg * neg_loss)
        else:
            loss = torch.sum(self.weight_pos * pos_loss + self.weight_neg * neg_loss) / pair_count

        if self.nu: loss = loss + beta_regularisation_loss.type(torch.cuda.FloatTensor)

        if self.alpha_gor > 0:

            # compute first moment (vector-wise dot product)
            # corr_a_n = torch.mm(batch[sampled_triplets[:, 0], :], batch[sampled_triplets[:, 2], :])
            n, m = batch.size(0), batch.size(1)
            idx = torch.from_numpy(np.array([[trip[0], trip[2]] for trip in sampled_triplets])).detach()
            corr_a_n = torch.bmm(batch[idx[:, 0], :].view(n, 1, m),
                                 batch[idx[:, 1], :].view(n, m, 1)).squeeze(1).squeeze(1)
            corr_mean = torch.pow(torch.mean(corr_a_n), 2)
            corr_var = torch.mean(torch.pow(corr_a_n, 2))
            gor = self.alpha_gor * (corr_mean + torch.nn.functional.relu(corr_var - (1/m)))

            loss = loss + gor
        else:
            corr_mean = -1
            corr_var = -1

        return loss, sampled_triplets, corr_mean, corr_var


"""================================================================================================="""
### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class MarginLoss_noise_constrainPos(torch.nn.Module):
    def __init__(self, margin=0.2, nu=0, beta=1.2, n_classes=100, beta_constant=False, sampling_method='distance_noise_constrainPos', weight_neg=1.0, weight_pos=1.0, pos_level=4):
        """
        Args:
            margin:             Triplet Margin.
            nu:                 Regularisation Parameter for beta values if they are learned.
            beta:               Class-Margin values.
            n_classes:          Number of different classes during training.
        """
        super(MarginLoss_noise_constrainPos, self).__init__()
        self.margin             = margin
        self.n_classes          = n_classes
        self.beta_constant     = beta_constant
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg
        self.pos_level = pos_level

        self.beta = beta if beta_constant else torch.nn.Parameter(torch.ones(n_classes)*beta)

        self.nu                 = nu

        self.sampler            = Sampler(method=sampling_method)


    def forward(self, batch, labels, gt_labels, sampled_triplets=None):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            gt_labels = gt_labels.detach().cpu().numpy()

        ###Setup for (learnable) class margin beta.
        if self.nu: beta_regularisation_loss = self.nu*torch.sum(self.beta)

        if sampled_triplets is None:
            sampled_triplets = self.sampler.give(batch, labels, gt_labels, pos_level=self.pos_level)

        d_ap = torch.stack([((batch[triplet[0],:]-batch[triplet[1],:]).pow(2).sum()+1e-8).pow(1/2) for triplet in sampled_triplets])
        d_an = torch.stack([((batch[triplet[0],:]-batch[triplet[2],:]).pow(2).sum()+1e-8).pow(1/2) for triplet in sampled_triplets])

        if self.beta_constant:
            beta = self.beta
        else:
            beta = torch.stack([self.beta[labels[triplet[0]]] for triplet in sampled_triplets]).type(torch.cuda.FloatTensor)

        pos_loss = torch.nn.functional.relu(d_ap-beta+self.margin)
        neg_loss = torch.nn.functional.relu(beta-d_an+self.margin)

        ###TODO: Adjust to cuda float
        # pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).type(torch.FloatTensor)
        pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).type(torch.cuda.FloatTensor)

        # if pair_count == 0.:
        #     loss = torch.sum(pos_loss+neg_loss)
        # else:
        #     loss = torch.sum(pos_loss+neg_loss)/pair_count

        if pair_count == 0.:
            loss = torch.sum(self.weight_pos * pos_loss + self.weight_neg * neg_loss)
        else:
            loss = torch.sum(self.weight_pos * pos_loss + self.weight_neg * neg_loss) / pair_count

        if self.nu: loss = loss + beta_regularisation_loss.type(torch.cuda.FloatTensor)

        return loss, sampled_triplets


"""================================================================================================="""
### Mixup regression loss.
class MixupRegression(torch.nn.Module):

    def __init__(self, norm='l2'):
        super(MixupRegression, self).__init__()

        if norm == 'l2':
            self.mse = torch.nn.MSELoss()
        elif norm == 'l1':
            self.mse = torch.nn.L1Loss()
        else:
            raise Exception('Unknown regression function !')

    def mix_embeds(self, raw_embed, lam, mix_indices):
        mix_embed_target = raw_embed * lam + raw_embed[mix_indices] * (1 - lam)
        mix_embed_target = torch.nn.functional.normalize(mix_embed_target, dim=-1) # normalize!

        return mix_embed_target


    def forward(self, mix_embed, raw_embed, lam, mix_indices):

        # create target embed locations
        mix_embed_target = self.mix_embeds(raw_embed, lam, mix_indices)

        # regress target positions
        loss = self.mse(mix_embed, mix_embed_target.detach())

        return loss


"""================================================================================================="""
class GradRev(torch.autograd.Function):
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -1.)

def grad_reverse(x):
    return GradRev()(x)

### Adversarial Loss
class AdvLoss(torch.nn.Module):
    def __init__(self, opt, classembd, clusterembd,proj_dim=512,lam_w=1e-2):
        ### Set Lambda to 1e-3
        super(AdvLoss, self).__init__()

        self.detach_target = opt.adv_detach_target
        self.pars        = opt
        self.classembd   = classembd
        self.clusterembd = clusterembd
        self.lam_w       = lam_w
        self.adv_dir = opt.adv_dir
        if proj_dim==111:
            self.regressor   = torch.nn.Sequential(
                torch.nn.Linear(clusterembd, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, classembd)).type(torch.cuda.FloatTensor)
        else:
            self.regressor   = torch.nn.Sequential(
                torch.nn.Linear(clusterembd, proj_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(proj_dim, classembd)).type(torch.cuda.FloatTensor)


    def forward(self, classfeatures, clusterfeatures, model=None):
        ### WITH Rev. Grad. Layer:
        if not self.pars.no_adv_class_reverse:
            classfeatures   = grad_reverse(classfeatures)
        if not self.pars.no_adv_intra_reverse:
            clusterfeatures = grad_reverse(clusterfeatures)

        if self.adv_dir == 'res_class':
            if self.detach_target:
                sim_loss = torch.mean(torch.mean((classfeatures.detach()*torch.nn.functional.normalize(self.regressor(clusterfeatures),dim=-1))**2,dim=-1))
            else:
                sim_loss = torch.mean(torch.mean((classfeatures*torch.nn.functional.normalize(self.regressor(clusterfeatures),dim=-1))**2,dim=-1))
        elif self.adv_dir == 'class_res':
            if self.detach_target:
                sim_loss = torch.mean(torch.mean((clusterfeatures.detach()*torch.nn.functional.normalize(self.regressor(classfeatures),dim=-1))**2,dim=-1)) # reverse direction
            else:
                sim_loss = torch.mean(torch.mean((clusterfeatures*torch.nn.functional.normalize(self.regressor(classfeatures),dim=-1))**2,dim=-1)) # reverse direction


        # reg_loss = 0
        # for mod in self.regressor:
        #     if isinstance(mod, nn.Linear):
        #         reg_loss = reg_loss + torch.nn.functional.relu(0,mod.bias.t().mm(mod.bias)-1) torch.sum((torch.sum(mod.weight*mod.weight(),dim=-1)-1)**2)
        # reg_loss = reg_loss + torch.sum((torch.sum(model.last_linear.weight*mod.last_linear.weight,dim=-1)-1)**2)
        # return -1.*sim_loss + self.lam_w*reg_loss
        return -1*sim_loss


### MI critic
def repeat_dense(input_dim, n_layers, n_units=512):
    """
    Repeat linear layers, attached to given input.
    """
    if n_units < 1 or n_layers < 1:
        raise ValueError('`n_layers` and `n_units` must be >= 1, '
                         'found {} and {} respectively'.format(n_layers, n_units))
    n_units = int(n_units)
    n_layers = int(n_layers)

    # create network
    h = list()
    h.append(torch.nn.Linear(input_dim, n_units))
    for i in range(1, n_layers):
        h.append(torch.nn.ReLU(inplace=True))
        h.append(torch.nn.Linear(n_units, n_units))

    return torch.nn.Sequential(*h)

    
class MI_Critic(torch.nn.Module):

    def __init__(self, phi_dim, n_layer=3, n_units=512, concat_input=False):
        super(MI_Critic, self).__init__()

        self.phi_dim = phi_dim
        self.concat_input = concat_input
        if not concat_input:
            self.class_stream = repeat_dense(phi_dim, n_layers=n_layer, n_units=n_units)
            self.res_stream = repeat_dense(phi_dim, n_layers=n_layer, n_units=n_units)
        else:
            self.comb_stream = repeat_dense(phi_dim * 2, n_layers=n_layer, n_units=n_units)
            self.score = torch.nn.Sequential(*[torch.nn.ReLU(inplace=True), torch.nn.Linear(n_units, 1)])

    def forward(self, phi_class, phi_res):

        if not self.concat_input:
            x_class = self.class_stream(phi_class)
            x_res = self.class_stream(phi_res)

            # vector-wise dot product
            x = torch.bmm(x_class.view(x_class.size(0), 1, x_class.size(1)), x_res.view(x_res.size(0), x_res.size(1), 1)).squeeze(1).squeeze(1)
        else:
            x = self.comb_stream(torch.cat((phi_class, phi_res), dim=1))
            x = self.score(x)

        return x


"""================================================================================================="""
### ProxyNCALoss containing trainable class proxies. Works independent of batch size.
class ProxyNCALoss(torch.nn.Module):
    def __init__(self, num_proxies, embedding_dim):
        """
        Implementation for PROXY-NCA loss. Note that here, Proxies are part of the loss, not the network.
        Args:
            num_proxies:    num_proxies is the number of Class Proxies to use. Normally equals the number of classes.
            embedding_dim:  Feature dimensionality. Proxies share the same dim. as the embeddings.
        """
        super(ProxyNCALoss, self).__init__()
        self.num_proxies   = num_proxies
        self.embedding_dim = embedding_dim
        self.PROXIES = torch.nn.Parameter(torch.randn(num_proxies, self.embedding_dim) / 8)
        self.all_classes = torch.arange(num_proxies)


    def forward(self, anchor_batch, classes):
        """
        Args:
            anchor_batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            classes: nparray/list: For each element of the anchor_batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        anchor_batch = 3*torch.nn.functional.normalize(anchor_batch, dim=1)
        PROXIES      = 3*torch.nn.functional.normalize(self.PROXIES, dim=1)
        pos_proxies = torch.stack([PROXIES[pos_label:pos_label+1,:] for pos_label in classes])
        neg_proxies = torch.stack([torch.cat([self.all_classes[:class_label],self.all_classes[class_label+1:]]) for class_label in classes])
        neg_proxies = torch.stack([PROXIES[neg_labels,:] for neg_labels in neg_proxies])

        dist_to_neg_proxies = torch.sum((anchor_batch[:,None,:]-neg_proxies).pow(2),dim=-1)
        dist_to_pos_proxies = torch.sum((anchor_batch[:,None,:]-pos_proxies).pow(2),dim=-1)

        negative_log_proxy_nca_loss = torch.mean(dist_to_pos_proxies[:,0] + torch.logsumexp(-dist_to_neg_proxies, dim=1))
        return negative_log_proxy_nca_loss



"""================================================================================================="""
### Container to use with latent space separation
def cluster(opt, dataloader, model, mode='mean_std', feat='embed_res', plot=False, verbose=False, epoch=-1):
    #Compute features
    _ = model.eval()
    with torch.no_grad():
        features = []
        final_iter  = tqdm(dataloader, desc='Computing Embeddings')
        for idx,inp in enumerate(final_iter):
            input_img = inp[-1]
            out = model(input_img.to(opt.device), feat=feat)

            if feat == 'embed_res':
                out = out[:,opt.classembed:]

            features.extend(out.cpu().detach().numpy().tolist())

        features = np.vstack(features).astype('float32')

    # Compute means and Center features
    classes = np.array([x[-1] for x in dataloader.dataset.image_list])
    if mode is not None:
        if 'mean' in mode:
            means   = np.stack([np.mean(features[classes==i,:],axis=0) for i in np.unique(classes)])
        if 'std' in mode:
            stds = np.stack([np.std(features[classes==i,:],axis=0) for i in np.unique(classes)])

        for i in range(len(features)):
            if 'mean' in mode:
                features[i,:] -= means[classes[i],:]
            if 'std' in mode:
                features[i,:] /= stds[classes[i],:]

    #Cluster
    n_samples, dim  = features.shape
    kmeans          = faiss.Kmeans(dim, opt.num_cluster)
    kmeans.n_iter, kmeans.min_points_per_centroid, kmeans.max_points_per_centroid = 20,5,1000000000

    kmeans.train(features)
    _, cluster_assignments = kmeans.index.search(features,1)

    return cluster_assignments


def random_cluster(opt, dataloader):
    cluster_assignments = np.random.choice(opt.num_cluster, len(dataloader.dataset))

    return cluster_assignments

def noise_cluster(opt, dataloader, noise_ratio=1.0):
    cluster_assignments = np.random.choice(opt.num_cluster, len(dataloader.dataset))

    return cluster_assignments

def find_knn(opt, dataloader, model, num_cluster=30, mode='mstd', init_generation=True):
    #Compute features
    _ = model.eval()
    with torch.no_grad():
        features = []
        final_iter  = tqdm(dataloader, desc='Computing Embeddings')
        for idx,inp in enumerate(final_iter):
            input_img = inp[-1]

            out = model(input_img.to(opt.device), is_cluster_generation=init_generation)
            if not init_generation: out = out[:,opt.classembed:]
            features.extend(out.cpu().detach().numpy().tolist())

        features = np.vstack(features).astype('float32')

    #Compute means and Center features
    classes = np.array([x[-1] for x in dataloader.dataset.image_list])
    if init_generation:
        means   = np.stack([np.mean(features[classes==i,:],axis=0) for i in np.unique(classes)])
        if mode=='mstd':  stds = np.stack([np.std(features[classes==i,:],axis=0) for i in np.unique(classes)])
        for i in range(len(features)):
            features[i,:] -= means[classes[i],:]
            if mode=='mstd': features[i,:] /= stds[classes[i],:]
        # else:
        #     for classnum in range(opt.num_classes):
        #         sel = classes==classnum
        #         X   = features[sel,:]
        #         pca = PCA(whiten=True)
        #         pca.fit(X)
        #         features[sel] = pca.inverse_transform(pca.transform(X))

    #Find Knn
    n_closest = 20
    n_samples, dim  = features.shape
    faiss_search_index = faiss.IndexFlatL2(dim)
    faiss_search_index.add(features)
    _, closest_feature_idxs = faiss_search_index.search(features, n_closest * 10)

    # remove samples from same class
    knns = [[idx for idx in knn if classes[idx] != classes[knn[0]]] for knn in closest_feature_idxs]
    knns = [k[0:min(n_closest, len(k))] for k in knns]

    # compute label variance
    knns_labels = [[classes[idx] for idx in knns[k]] for k in range(n_samples)]
    knns_variance = [len(np.unique(label)) / n_closest  for label in knns_labels]
    print('*** Mean variance knns: {}'.format(round(np.mean(knns_variance), 3)))

    return knns


"""================= HELPER ==================================="""
def analyse_data_distribution(opt, dataloader, model, mode='mean_std', feat='embed_res'):
    #Compute features
    _ = model.eval()
    with torch.no_grad():
        features = []
        final_iter  = tqdm(dataloader, desc='Computing Embeddings')
        for idx,inp in enumerate(final_iter):
            input_img = inp[-1]
            out = model(input_img.to(opt.device), feat=feat)

            if feat == 'embed_res':
                out = out[:,opt.classembed:]

            features.extend(out.cpu().detach().numpy().tolist())

        features = np.vstack(features).astype('float32')

    # Compute means and Center features
    classes = np.array([x[-1] for x in dataloader.dataset.image_list])
    if mode is not None:
        if 'mean' in mode:
            means   = np.stack([np.mean(features[classes==i,:],axis=0) for i in np.unique(classes)])
        if 'std' in mode:
            stds = np.stack([np.std(features[classes==i,:],axis=0) for i in np.unique(classes)])

        for i in range(len(features)):
            if 'mean' in mode:
                features[i,:] -= means[classes[i],:]
            if 'std' in mode:
                features[i,:] /= stds[classes[i],:]

    # #Kmeans
    # n_samples, dim = features.shape
    # n_cluster = list(range(10, 500, 10))
    # error_per_k = list()
    # for k in n_cluster:
    #     kmeans          = faiss.Kmeans(dim, k)
    #     kmeans.n_iter, kmeans.min_points_per_centroid, kmeans.max_points_per_centroid = 20,5,1000000000
    #
    #     kmeans.train(features)
    #     D, _ = kmeans.index.search(features,1)
    #     error_per_k.append(np.sum(D))

    #PCA
    from sklearn.decomposition import PCA
    n_pca_dims = 100
    pca = PCA(n_components=n_pca_dims)
    feat_pca = pca.fit_transform(features)
    energy_per_k = [np.sum(pca.explained_variance_ratio_[0:k]) for k in range(n_pca_dims)]

    # project to axis, sort and plot
    for i in range(n_pca_dims):

        # project onto top comps
        feat_project = feat_pca[:, i]

        # sort samples w.r.t. projection
        ids_sort = np.argsort(feat_project)
        # steps = np.linspace(0, features.shape[0], num=100, dtype=int, endpoint=False)

        n_steps = features.shape[0] // 100
        for j in range(n_steps):
            steps = list(range(j * 100, (j+1) * 100))
            samples_to_plot = [ids_sort[k] for k in steps]

            # plot sorted samples
            visualize_single_cluster(samples_to_plot, dataloader.dataset.image_list,
                                     subtitles=None, n_max_samples=100,
                                     path_save=None, plot=True,
                                     title='axis {}, expl. var. = {}'.format(i, pca.explained_variance_ratio_[i]))

    # # joint summary
    # import matplotlib.pyplot as plt
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # f.suptitle('pooled layer4 + pooled layer2 (non-standardized)') # + pooled layer4
    #
    # ax1.plot(n_cluster, error_per_k)
    # ax1.set(xlabel='n_cluster', ylabel='l2 error')
    # ax1.set_title('Kmeans error')
    #
    # ax2.plot(energy_per_k)
    # ax2.set(xlabel='n_components', ylabel='accumulated energy')
    # ax2.set_title('PCA energy')
    # plt.show(False)
    # f.savefig('./Analysis/lay2_lay4_nonstand.pdf') # lay4_


def visualize_single_cluster(cluster, dataset, subtitles=None, n_max_samples=100, path_save=None, plot=False, title=None, close_fig=True):
    import matplotlib.pyplot as plt

    # visualize cluster
    n_max_row = 10
    members = np.asarray(cluster)
    members = members[0:min(n_max_samples, len(members))]  # cut cluster if needed
    img_paths = [dataset[k][0] for k in members]

    # visualize samples
    n_cols = len(members) // n_max_row + int(len(members) % n_max_row > 0)
    fig, axes = plt.subplots(max(n_cols, 2), n_max_row)
    if title is not None: fig.suptitle('{}'.format(title), fontsize=16)

    count = 0
    stop_plot = False
    for i in range(n_cols):
        if stop_plot: break

        for j in range(n_max_row):
            img = plt.imread(img_paths[count])
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            if subtitles is not None:
                axes[i, j].set_title('{}'.format(subtitles[count]), color=(0.0, 1.0, 0.0))
            count += 1

            if count == len(members):
                stop_plot = True
                break

    fig.tight_layout()
    if plot:
        plt.show(True)
    else:
        plt.show(False)

    if path_save is not None:
        fig.savefig(path_save, bbox_inches='tight', dpi=300)

    if close_fig:
        plt.close()


def compute_cluster_purity(cluster_ids, gt_labels):
    member_labels = [gt_labels[k] for k in cluster_ids]
    cluster_label = max(set(member_labels), key=member_labels.count)
    purity = member_labels.count(cluster_label) / len(cluster_ids)

    return purity, member_labels, cluster_label

