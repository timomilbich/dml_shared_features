"""============================================================================================================="""
######## LIBRARIES #####################
import warnings
warnings.filterwarnings("ignore")

import numpy as np, os, sys, pandas as pd, csv
import torch, torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import faiss
from sklearn import metrics
from sklearn import cluster
from sklearn.metrics import pairwise_distances

import datetime
import pickle as pkl



"""============================================================================================================="""
################# ACQUIRE NUMBER OF WEIGHTS #################
def gimme_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


################# SAVE TRAINING PARAMETERS IN NICE STRING #################
def gimme_save_string(opt):
    varx = vars(opt)
    base_str = ''
    for key in varx:
        base_str += str(key)
        if isinstance(varx[key],dict):
            for sub_key, sub_item in varx[key].items():
                base_str += '\n\t'+str(sub_key)+': '+str(sub_item)
        else:
            base_str += '\n\t'+str(varx[key])
        base_str+='\n\n'
    return base_str



"""============================================================================================================="""
def eval_metrics(model, test_dataloader, device, k_vals=[1,2,4,8], spliteval=False, weight_class=None, weight_cluster=None, epoch=0, opt=None, embed_type='class'):
    torch.cuda.empty_cache()

    _ = model.eval()
    n_classes = len(test_dataloader.dataset.avail_classes)

    ### For all test images, extract features
    with torch.no_grad():
        target_labels, feature_coll = [],[]
        final_iter = tqdm(test_dataloader, desc='Computing Evaluation Metrics...')
        image_paths= [x[0] for x in test_dataloader.dataset.image_list]
        for idx,inp in enumerate(final_iter):
            input_img,target = inp[-1], inp[0]
            target_labels.extend(target.numpy().tolist())
            out = model(input_img.to(device))
            if spliteval:
                if embed_type == 'class':
                    out = out[:,:opt.classembed]
                elif embed_type == 'res':
                    out = out[:,opt.classembed:]
                else:
                    raise Exception('Unknown embed type!')
            # else:
            #     if weight_class is not None and weight_cluster is not None:
            #         out[:,:opt.classembed] *= weight_class
            #         out[:,opt.classembed:] *= weight_cluster
            #         out = torch.nn.functional.normalize(out, dim=-1)
            feature_coll.extend(out.cpu().detach().numpy().tolist())

        target_labels = np.hstack(target_labels).reshape(-1,1)
        feature_coll  = np.vstack(feature_coll)
        ### TODO CHECK EVAL PART HERE
        # feature_coll  = (feature_coll-np.min(feature_coll))/(np.max(feature_coll)-np.min(feature_coll))
        feature_coll  = feature_coll.astype('float32')

        torch.cuda.empty_cache()
        ### Set CPU Cluster index
        cpu_cluster_index = faiss.IndexFlatL2(feature_coll.shape[-1])
        kmeans            = faiss.Clustering(feature_coll.shape[-1], n_classes)
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
        NMI = metrics.cluster.normalized_mutual_info_score(model_generated_cluster_labels.reshape(-1), target_labels.reshape(-1))


        ### Recover max(k_vals) nearest neighbours to use for recall computation
        faiss_search_index  = faiss.IndexFlatL2(feature_coll.shape[-1])
        faiss_search_index.add(feature_coll)
        ### NOTE: when using the same array for search and base, we need to ignore the first returned element.
        _, k_closest_points = faiss_search_index.search(feature_coll, int(np.max(k_vals)+1))
        k_closest_classes   = target_labels.reshape(-1)[k_closest_points[:,1:]]

        ### Compute Recall
        recall_all_k = []
        for k in k_vals:
            recall_at_k = np.sum([1 for target, recalled_predictions in zip(target_labels, k_closest_classes) if target in recalled_predictions[:k]])/len(target_labels)
            recall_all_k.append(recall_at_k)
        recall_str = ', '.join('@{0}: {1:.4f}'.format(k,rec) for k,rec in zip(k_vals, recall_all_k))

        message = 'Epoch (Test) {0}: NMI [{1:.4f}] | Recall [{2}]'.format(epoch, NMI, recall_str)
        print(message)

        # compute mean intra-class and inter-class distance
        mean_intra_dist = -1
        mean_inter_dist = -1
        if epoch % opt.freq_analyse_inter_intra_dist == 0:
            # compute pairwise distances
            all_dists = pairwise_distances(feature_coll, metric='euclidean')

            # compute intra distances
            intra_dists_all = list()
            inter_dists_all = list()
            for k in np.unique(target_labels):
                # intra
                ids_member = np.where(target_labels == k)[0]
                dists_tmp = all_dists[ids_member, :][:, ids_member]
                intra_dists_all.append(np.mean(dists_tmp))

                # inter
                dists_tmp = all_dists
                dists_tmp[ids_member[0]:ids_member[-1]+1, ids_member[0]:ids_member[-1]+1] = -1
                dists_tmp = dists_tmp.flatten()
                dists_tmp = dists_tmp[np.where(dists_tmp != -1)[0]]
                inter_dists_all.append(np.mean(dists_tmp))

            mean_intra_dist = np.mean(intra_dists_all)
            mean_inter_dist = np.mean(inter_dists_all)

    return NMI, recall_all_k, feature_coll, image_paths, mean_intra_dist, mean_inter_dist


def eval_metrics_inshop(model, query_dataloader, gallery_dataloader, device, k_vals=[1,10,20,30,50], spliteval=False, weight_class=None, weight_cluster=None, epoch=0, opt=None):
    torch.cuda.empty_cache()

    _ = model.eval()
    n_classes = len(query_dataloader.dataset.avail_classes)

    ### For all test images, extract features
    with torch.no_grad():
        query_target_labels, query_feature_coll     = [],[]
        query_image_paths   = [x[0] for x in query_dataloader.dataset.image_list]
        query_iter = tqdm(query_dataloader, desc='Extraction Query Features')
        for idx,inp in enumerate(query_iter):
            input_img,target = inp[-1], inp[0]
            query_target_labels.extend(target.numpy().tolist())
            out = model(input_img.to(device))
            if spliteval: out = out[:,:opt.classembed]
            else:
                if weight_class is not None and weight_cluster is not None:
                    out[:,:opt.classembed] *= weight_class
                    out[:,opt.classembed:] *= weight_cluster
                    out = torch.nn.functional.normalize(out, dim=-1)
            query_feature_coll.extend(out.cpu().detach().numpy().tolist())

        gallery_target_labels, gallery_feature_coll = [],[]
        gallery_image_paths = [x[0] for x in gallery_dataloader.dataset.image_list]
        gallery_iter = tqdm(gallery_dataloader, desc='Extraction Gallery Features')
        for idx,inp in enumerate(gallery_iter):
            input_img,target = inp[-1], inp[0]
            gallery_target_labels.extend(target.numpy().tolist())
            out = model(input_img.to(device))
            if spliteval: out = out[:,:opt.classembed]
            else:
                if weight_class is not None and weight_cluster is not None:
                    out[:,:opt.classembed] *= weight_class
                    out[:,opt.classembed:] *= weight_cluster
                    out = torch.nn.functional.normalize(out, dim=-1)
            gallery_feature_coll.extend(out.cpu().detach().numpy().tolist())


        query_target_labels, query_feature_coll     = np.hstack(query_target_labels).reshape(-1,1), np.vstack(query_feature_coll).astype('float32')
        gallery_target_labels, gallery_feature_coll = np.hstack(gallery_target_labels).reshape(-1,1), np.vstack(gallery_feature_coll).astype('float32')

        torch.cuda.empty_cache()

        #################### COMPUTE NMI #######################
        ### Set CPU Cluster index
        stackset    = np.concatenate([query_feature_coll, gallery_feature_coll],axis=0)
        stacklabels = np.concatenate([query_target_labels, gallery_target_labels],axis=0)
        cpu_cluster_index = faiss.IndexFlatL2(stackset.shape[-1])
        kmeans            = faiss.Clustering(stackset.shape[-1], n_classes)
        kmeans.niter = 20
        kmeans.min_points_per_centroid = 1
        kmeans.max_points_per_centroid = 1000000000

        ### Train Kmeans
        kmeans.train(stackset, cpu_cluster_index)
        computed_centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes, stackset.shape[-1])

        ### Assign feature points to clusters
        faiss_search_index = faiss.IndexFlatL2(computed_centroids.shape[-1])
        faiss_search_index.add(computed_centroids)
        _, model_generated_cluster_labels = faiss_search_index.search(stackset, 1)

        ### Compute NMI
        NMI = metrics.cluster.normalized_mutual_info_score(model_generated_cluster_labels.reshape(-1), stacklabels.reshape(-1))

        ### Recover max(k_vals) nearest neighbours to use for recall computation
        faiss_search_index  = faiss.IndexFlatL2(gallery_feature_coll.shape[-1])
        faiss_search_index.add(gallery_feature_coll)
        _, k_closest_points = faiss_search_index.search(query_feature_coll, int(np.max(k_vals)))
        k_closest_classes   = gallery_target_labels.reshape(-1)[k_closest_points]

        ### Compute Recall
        recall_all_k = []
        for k in k_vals:
            recall_at_k = np.sum([1 for target, recalled_predictions in zip(query_target_labels, k_closest_classes) if target in recalled_predictions[:k]])/len(query_target_labels)
            recall_all_k.append(recall_at_k)
        recall_str = ', '.join('@{0}: {1:.4f}'.format(k,rec) for k,rec in zip(k_vals, recall_all_k))


        message = 'Epoch (Test) {0}: NMI [{1:.4f}] | Recall [{2}]'.format(epoch, NMI, recall_str)
        print(message)


    return NMI, recall_all_k, query_feature_coll, gallery_feature_coll, query_image_paths, gallery_image_paths



"""============================================================================================================="""
####### RECOVER CLOSEST EXAMPLE IMAGES #######
def recover_closest(feature_list, image_paths, save_path, n_image_samples=10, n_closest=3):
    image_paths = np.array(image_paths)
    sample_idxs = np.random.choice(np.arange(len(feature_list)), n_image_samples)

    faiss_search_index = faiss.IndexFlatL2(feature_list.shape[-1])
    faiss_search_index.add(feature_list)
    _, closest_feature_idxs = faiss_search_index.search(feature_list, n_closest+1)

    sample_paths = image_paths[closest_feature_idxs][sample_idxs]

    f,axes = plt.subplots(n_image_samples, n_closest+1)
    for i,(ax,plot_path) in enumerate(zip(axes.reshape(-1), sample_paths.reshape(-1))):
        ax.imshow(np.array(Image.open(plot_path)))
        ax.set_xticks([])
        ax.set_yticks([])
        if i%(n_closest+1):
            ax.axvline(x=0, color='g', linewidth=13)
        else:
            ax.axvline(x=0, color='r', linewidth=13)
    f.set_size_inches(10,20)
    f.tight_layout()
    f.savefig(save_path)
    plt.close()


####### RECOVER CLOSEST EXAMPLE IMAGES #######
def recover_closest_inshop(query_feature_list, gallery_feature_list, query_image_paths, gallery_image_paths, save_path, n_image_samples=10, n_closest=3):
    query_image_paths   = np.array(query_image_paths)
    gallery_image_paths = np.array(gallery_image_paths)
    sample_idxs = np.random.choice(np.arange(len(gallery_feature_list)), n_image_samples)

    faiss_search_index = faiss.IndexFlatL2(gallery_feature_list.shape[-1])
    faiss_search_index.add(gallery_feature_list)
    _, closest_feature_idxs = faiss_search_index.search(query_feature_list, n_closest)

    image_paths  = gallery_image_paths[closest_feature_idxs]
    image_paths  = np.concatenate([query_image_paths, gallery_image_paths],axis=-1)

    sample_paths = image_paths[closest_feature_idxs][sample_idxs]

    f,axes = plt.subplots(n_image_samples, n_closest+1)
    for i,(ax,plot_path) in enumerate(zip(axes.reshape(-1), sample_paths.reshape(-1))):
        ax.imshow(np.array(Image.open(plot_path)))
        ax.set_xticks([])
        ax.set_yticks([])
        if i%(n_closest+1):
            ax.axvline(x=0, color='g', linewidth=13)
        else:
            ax.axvline(x=0, color='r', linewidth=13)
    f.set_size_inches(10,20)
    f.tight_layout()
    f.savefig(save_path)
    # plt.show()
    plt.close()



"""============================================================================================================="""
################## SET NETWORK TRAINING CHECKPOINT #####################
def set_checkpoint(model, opt, epoch, optimizer, savepath, progress_saver, suffix=None):

    path_checkpoint = savepath+'/checkpoint'
    if suffix is not None: path_checkpoint += '_' + suffix
    torch.save({'epoch': epoch+1, 'state_dict':model.state_dict(),
                'optim_state_dict':optimizer.state_dict(), 'opt':opt,
                'progress':progress_saver}, path_checkpoint+'.pth.tar')


"""============================================================================================================="""
################## WRITE TO CSV FILE #####################
class CSV_Writer():
    def __init__(self, save_path, columns):
        self.save_path = save_path
        self.columns   = columns

        with open(self.save_path, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(self.columns)

    def log(self, inputs):
        with open(self.save_path, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(inputs)



################## PLOT SUMMARY IMAGE #####################
class InfoPlotter():
    def __init__(self, save_path, title='Training Log', figsize=(15,10)):
        self.save_path = save_path
        self.title     = title
        self.figsize   = figsize
        self.colors    = ['r','g','b','y','m', 'c']

    def make_plot(self, x, y1, y2s, labels=['Training', 'Validation']):
        plt.style.use('ggplot')
        f,ax = plt.subplots(1)
        ax.set_title(self.title)
        ax.plot(x, y1, '-k', label=labels[0])
        axx = ax.twinx()
        for i,y2 in enumerate(y2s):
            axx.plot(x, y2, '-{}'.format(self.colors[i]), label=labels[i+1])
        f.legend()
        f.set_size_inches(self.figsize[0], self.figsize[1])
        f.savefig(self.save_path)
        plt.close()


################## GENERATE LOGGING FOLDER/FILES #######################
def set_logging(opt):
    checkfolder = opt.save_path+'/'+opt.savename
    if opt.savename == '':
        date = datetime.datetime.now()
        time_string = '{}-{}-{}-{}-{}-{}'.format(date.year, date.month, date.day, date.hour, date.minute, date.second)
        checkfolder = opt.save_path+'/{}_{}_{}_{}_'.format(opt.dataset.upper(), opt.loss.upper(), opt.sampling.upper(), opt.arch.upper())+time_string
    # counter     = 1
    # while os.path.exists(checkfolder):
    #     checkfolder = opt.save_path+'_'+str(counter)
    #     counter += 1
    if not os.path.exists(checkfolder): os.makedirs(checkfolder)
    opt.save_path = checkfolder

    # with open(opt.save_path+'/Parameter_Info.txt','w') as f:
    #     f.write(gimme_save_string(opt))
    # pkl.dump(opt,open(opt.save_path+"/hypa.pkl","wb"))



################## GENERATE tSNE PLOTS #######################
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.preprocessing import StandardScaler
import cv2

def plot_tSNE(opt, dataloader, model, feat='embed_class', type_data='trainval', n_samples=100, img_size= 45, perplex=40.0, img_zoom=0.7, epoch=0):

    #Compute features
    _ = model.eval()
    with torch.no_grad():
        features_res = list()
        features_class = list()
        final_iter  = tqdm(dataloader, desc='Computing Embeddings')
        for idx,inp in enumerate(final_iter):
            input_img = inp[-1]

            if 'embed' in feat:
                out = model(input_img.to(opt.device), feat='embed')
                out_res = out[:,opt.classembed:]
                out_class = out[:,:opt.classembed]
            else:
                raise Exception('feature type not supported!')

            features_class.extend(out_class.cpu().detach().numpy().tolist())
            features_res.extend(out_res.cpu().detach().numpy().tolist())

        features_class = np.vstack(features_class).astype('float32')
        features_res = np.vstack(features_res).astype('float32')

    # choose subset
    idx2use = np.random.choice(features_class.shape[0], size=n_samples, replace=False)
    features_class_sub = features_class[idx2use, :]
    features_res_sub = features_res[idx2use, :]

    # prepare images
    image_paths = np.array([x[0] for x in dataloader.dataset.image_list])
    image_paths_sub = [image_paths[k] for k in idx2use]
    image_labels = np.array([x[1] for x in dataloader.dataset.image_list])
    image_labels_sub = [image_labels[k] for k in idx2use]

    images = []
    labels_unique = np.unique(image_labels_sub)
    cmap = plt.cm.get_cmap('hsv', len(labels_unique))
    top, bottom, left, right = [int(img_size * 0.05)] * 4
    for path, label in zip(image_paths_sub, image_labels_sub):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (img_size, img_size))
        c = cmap(np.where(labels_unique == label)[0][0])
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)])

        images.append(np.array(image))
    images = np.array(images)

    # plot
    save_path_base = opt.save_path + '/tSNE/epoch_{}'.format(epoch)
    if not os.path.isdir(save_path_base):
        os.makedirs(save_path_base)

    for p in perplex:
        ### SCATTER PLOTS

        # tsne class embedding
        save_path = save_path_base + '/scatter_class_{}_p{}.svg'.format(type_data, int(p))
        visualize_scatter(features_class, image_labels, p, save_path=save_path)

        # tsne residual embedding
        save_path = save_path_base + '/scatter_res_{}_p{}.svg'.format(type_data, int(p))
        visualize_scatter(features_res, image_labels, p, save_path=save_path)


        ### IMAGE PLOTS

        # tsne class embedding
        save_path = save_path_base + '/img_class_{}_p{}.svg'.format(type_data, int(p))
        visualize_tsne_with_images(features_class_sub, p, images=images, image_zoom=img_zoom, save_path=save_path)

        # tsne residual embedding
        save_path = save_path_base + '/img_res_{}_p{}.svg'.format(type_data, int(p))
        visualize_tsne_with_images(features_res_sub, p, images=images, image_zoom=img_zoom, save_path=save_path)


def visualize_tsne_with_images(features, perplex, images, figsize=(30,30), image_zoom=1, save_path=None):
    # compute tSNE
    tsne = TSNE(n_components=2, perplexity=perplex)
    tsne_result = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(tsne_result, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(tsne_result)
    ax.autoscale()
    plt.show(False)

    # save plot
    fig.savefig(save_path, dpi=1000)
    plt.close()


def visualize_scatter(features, img_labels, perplex, figsize=(10,10), save_path=None):

    # use subset
    ids2use = np.random.choice(len(img_labels), size=3000, replace=False)
    features = features[ids2use, :]
    img_labels = img_labels[ids2use]

    # compute tSNE
    tsne = TSNE(n_components=2, perplexity=perplex)
    tsne_result = tsne.fit_transform(features)

    # plot
    cmap = plt.cm.get_cmap('hsv', len(np.unique(img_labels)))
    fig, ax = plt.subplots(figsize=figsize)
    for id in range(len(img_labels)):
        plt.scatter(tsne_result[id, 0],  tsne_result[id, 1],
                    marker='o',
                    color=cmap(img_labels[id]),
                    linewidth='1',
                    alpha=0.8)
    ax.update_datalim(tsne_result)
    ax.autoscale()
    plt.show(False)

    # save plot
    fig.savefig(save_path, dpi=1000)
    plt.close()