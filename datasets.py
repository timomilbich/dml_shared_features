import warnings
warnings.filterwarnings("ignore")

import numpy as np, os, sys, pandas as pd, csv, copy
import torch, torch.nn as nn, matplotlib.pyplot as plt, random

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import pretrainedmodels.utils as utils

################# FUNCTIONS TO RETURN TRAIN/VAL PYTORCH DATASETS FOR CUB200, CARS196 AND STANFORD ONLINE PRODUCTS ####################################
def give_CUB200_datasets(opt):
    """
    This function generates a training and testing dataloader for Metric Learning on the CUB-200-2011 dataset.
    For Metric Learning, the dataset is sorted by name, and the first halt used for training while the last half is used for testing.
    So no random shuffling of classes.
    """
    image_sourcepath  = opt.source_path+'/images'
    image_classes = sorted([x for x in os.listdir(image_sourcepath) if '._' not in x], key=lambda x: int(x.split('.')[0]))
    conversion    = {int(x.split('.')[0]):x.split('.')[-1] for x in image_classes}
    image_list    = {int(key.split('.')[0]):sorted([image_sourcepath+'/'+key+'/'+x for x in os.listdir(image_sourcepath+'/'+key) if '._' not in x]) for key in image_classes}
    image_list    = [[(key,img_path) for img_path in image_list[key]] for key in image_list.keys()]
    image_list    = [x for y in image_list for x in y]

    image_dict    = {}
    for key, img_path in image_list:
        key = key-1
        if not key in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append(img_path)

    keys = sorted(list(image_dict.keys()))
    # random.shuffle(keys)
    #Following "Deep Metric Learning via Lifted Structured Feature Embedding", we use the first half of classes for training.
    train,test = keys[:len(keys)//2], keys[len(keys)//2:]

    train_image_dict, val_image_dict = {key:image_dict[key] for key in train},{key:image_dict[key] for key in test}


    train_split = BaseTripletDataset(train_image_dict, opt, samples_per_class=opt.samples_per_class)
    val_split   = BaseTripletDataset(val_image_dict,   opt, is_validation=True)
    eval_split  = BaseTripletDataset(train_image_dict, opt, is_validation=True)
    train_split.conversion = conversion
    val_split.conversion   = conversion
    eval_split.conversion   = conversion
    return train_split, val_split, eval_split


def give_CARS196_datasets(opt):
    """
    This function generates a training and testing dataloader for Metric Learning on the CARS-196 dataset.
    For Metric Learning, the dataset is sorted by name, and the first halt used for training while the last half is used for testing.
    So no random shuffling of classes.
    """
    image_sourcepath  = opt.source_path+'/images'
    image_classes = sorted([x for x in os.listdir(image_sourcepath)])
    conversion    = {i:x for i,x in enumerate(image_classes)}
    image_list    = {i:sorted([image_sourcepath+'/'+key+'/'+x for x in os.listdir(image_sourcepath+'/'+key)]) for i,key in enumerate(image_classes)}
    image_list    = [[(key,img_path) for img_path in image_list[key]] for key in image_list.keys()]
    image_list    = [x for y in image_list for x in y]

    image_dict    = {}
    for key, img_path in image_list:
        key = key-1
        if not key in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append(img_path)

    keys = sorted(list(image_dict.keys()))
    # random.shuffle(keys)
    #Following "Deep Metric Learning via Lifted Structured Feature Embedding", we use the first half of classes for training.
    train,test = keys[:len(keys)//2], keys[len(keys)//2:]
    train_image_dict, val_image_dict = {key:image_dict[key] for key in train},{key:image_dict[key] for key in test}

    train_split = BaseTripletDataset(train_image_dict, opt, samples_per_class=opt.samples_per_class)
    val_split   = BaseTripletDataset(val_image_dict,   opt, is_validation=True)
    eval_split  = BaseTripletDataset(train_image_dict, opt, is_validation=True)
    train_split.conversion = conversion
    val_split.conversion   = conversion
    eval_split.conversion   = conversion
    return train_split, val_split, eval_split


def give_OnlineProducts_datasets(opt):
    image_sourcepath  = opt.source_path+'/images'
    training_files = pd.read_table(opt.source_path+'/Info_Files/Ebay_train.txt', header=0, delimiter=' ')
    test_files     = pd.read_table(opt.source_path+'/Info_Files/Ebay_test.txt', header=0, delimiter=' ')


    conversion, super_conversion = {},{}
    for class_id, path in zip(training_files['class_id'],training_files['path']):
        conversion[class_id] = path.split('/')[0]
    for super_class_id, path in zip(training_files['super_class_id'],training_files['path']):
        conversion[super_class_id] = path.split('/')[0]
    for class_id, path in zip(test_files['class_id'],test_files['path']):
        conversion[class_id] = path.split('/')[0]

    train_image_dict, val_image_dict, super_train_image_dict  = {},{},{}
    for key, img_path in zip(training_files['class_id'],training_files['path']):
        key = key-1
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(image_sourcepath+'/'+img_path)

    for key, img_path in zip(test_files['class_id'],test_files['path']):
        key = key-1
        if not key in val_image_dict.keys():
            val_image_dict[key] = []
        val_image_dict[key].append(image_sourcepath+'/'+img_path)

    for key, img_path in zip(training_files['super_class_id'],training_files['path']):
        key = key-1
        if not key in super_train_image_dict.keys():
            super_train_image_dict[key] = []
        super_train_image_dict[key].append(image_sourcepath+'/'+img_path)


    train_split = BaseTripletDataset(train_image_dict, opt, samples_per_class=opt.samples_per_class)
    super_train_split = BaseTripletDataset(super_train_image_dict, opt, is_validation=True)
    val_split   = BaseTripletDataset(val_image_dict,   opt, is_validation=True)
    eval_split  = BaseTripletDataset(train_image_dict, opt, is_validation=True)
    train_split.conversion = conversion
    super_train_split.conversion = super_conversion
    val_split.conversion   = conversion
    eval_split.conversion   = conversion
    return train_split, val_split, eval_split, super_train_split


def give_InShop_datasets(opt):
    data_info = np.array(pd.read_table(opt.source_path+'/Eval/list_eval_partition.txt', header=1, delim_whitespace=True))[1:,:]
    train, query, gallery   = data_info[data_info[:,2]=='train'][:,:2], data_info[data_info[:,2]=='query'][:,:2], data_info[data_info[:,2]=='gallery'][:,:2]
    lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in train[:,1]])))}
    train[:,1] = np.array([lab_conv[int(x.split('_')[-1])] for x in train[:,1]])
    lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in np.concatenate([query[:,1], gallery[:,1]])])))}
    query[:,1]   = np.array([lab_conv[int(x.split('_')[-1])] for x in query[:,1]])
    gallery[:,1] = np.array([lab_conv[int(x.split('_')[-1])] for x in gallery[:,1]])

    train_image_dict    = {}
    for img_path, key in train:
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(opt.source_path+'/'+img_path)

    query_image_dict    = {}
    for img_path, key in query:
        if not key in query_image_dict.keys():
            query_image_dict[key] = []
        query_image_dict[key].append(opt.source_path+'/'+img_path)

    gallery_image_dict    = {}
    for img_path, key in gallery:
        if not key in gallery_image_dict.keys():
            gallery_image_dict[key] = []
        gallery_image_dict[key].append(opt.source_path+'/'+img_path)

    train_dataset = BaseTripletDataset(train_image_dict, opt, samples_per_class=opt.samples_per_class)
    eval_dataset  = BaseTripletDataset(train_image_dict, opt, is_validation=True)
    query_dataset     = BaseTripletDataset(query_image_dict, opt, is_validation=True)
    gallery_dataset   = BaseTripletDataset(gallery_image_dict, opt, is_validation=True)
    return train_dataset, eval_dataset, query_dataset, gallery_dataset


def give_VehicleID_datasets(opt):
    train = np.array(pd.read_table(opt.source_path+'/train_test_split/train_list.txt', header=None, delim_whitespace=True))
    small_test  = np.array(pd.read_table(opt.source_path+'/train_test_split/test_list_800.txt', header=None, delim_whitespace=True))
    medium_test = np.array(pd.read_table(opt.source_path+'/train_test_split/test_list_1600.txt', header=None, delim_whitespace=True))
    big_test    = np.array(pd.read_table(opt.source_path+'/train_test_split/test_list_2400.txt', header=None, delim_whitespace=True))
    lab_conv = {x:i for i,x in enumerate(np.unique(train[:,1]))}
    train[:,1] = np.array([lab_conv[x] for x in train[:,1]])
    lab_conv = {x:i for i,x in enumerate(np.unique(np.concatenate([small_test[:,1], medium_test[:,1], big_test[:,1]])))}
    small_test[:,1]  = np.array([lab_conv[x] for x in small_test[:,1]])
    medium_test[:,1] = np.array([lab_conv[x] for x in medium_test[:,1]])
    big_test[:,1]    = np.array([lab_conv[x] for x in big_test[:,1]])

    train_image_dict    = {}
    for img_path, key in train:
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(opt.source_path+'/image/{:07d}.jpg'.format(img_path))

    small_test_dict = {}
    for img_path, key in small_test:
        if not key in small_test_dict.keys():
            small_test_dict[key] = []
        small_test_dict[key].append(opt.source_path+'/image/{:07d}.jpg'.format(img_path))

    medium_test_dict    = {}
    for img_path, key in medium_test:
        if not key in medium_test_dict.keys():
            medium_test_dict[key] = []
        medium_test_dict[key].append(opt.source_path+'/image/{:07d}.jpg'.format(img_path))

    big_test_dict    = {}
    for img_path, key in big_test:
        if not key in big_test_dict.keys():
            big_test_dict[key] = []
        big_test_dict[key].append(opt.source_path+'/image/{:07d}.jpg'.format(img_path))

    train_dataset = BaseTripletDataset(train_image_dict, opt, samples_per_class=opt.samples_per_class)
    eval_dataset  = BaseTripletDataset(train_image_dict, opt,    is_validation=True)
    val_small_dataset     = BaseTripletDataset(small_test_dict, opt,  is_validation=True)
    val_medium_dataset    = BaseTripletDataset(medium_test_dict, opt, is_validation=True)
    val_big_dataset       = BaseTripletDataset(big_test_dict, opt,    is_validation=True)
    return train_dataset, eval_dataset, val_small_dataset, val_medium_dataset, val_big_dataset




################## BASIC PYTORCH DATASET USED FOR ALL DATASETS ##################################
class BaseTripletDataset(Dataset):
    def __init__(self, image_dict, opt, samples_per_class=8, is_validation=False):
        self.n_files     = np.sum([len(image_dict[key]) for key in image_dict.keys()])

        self.is_validation = is_validation

        self.pars        = opt
        self.image_dict  = image_dict

        self.avail_classes    = sorted(list(self.image_dict.keys()))
        # self.n_class_in_batch = batch_size//samples_per_class
        # assert self.n_class_in_batch*samples_per_class == batch_size, "Provided batch size {} not divisible by {} samples per class".format(batch_size, samples_per_class)
        #Convert image dictionary from classname:content to class_idx:content
        self.image_dict    = {i:self.image_dict[key] for i,key in enumerate(self.avail_classes)}
        self.avail_classes = sorted(list(self.image_dict.keys()))

        self.cluster_labels = None

        if not self.is_validation:
            self.samples_per_class = samples_per_class
            #Select current class to sample images from up to <samples_per_class>
            self.current_class   = np.random.randint(len(self.avail_classes))
            self.classes_visited = [self.current_class, self.current_class]
            self.n_samples_drawn = 0


        ##### Option 2: Use Mean/Stds on which the networks were trained
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transf_list = []
        if not self.is_validation:
            transf_list.extend([#transforms.RandomRotation(30),
                                #transforms.Resize(256),
                                transforms.RandomResizedCrop(size=224),
                                #transforms.RandomCrop(224),
                                transforms.RandomHorizontalFlip(0.5)])
        else:
            transf_list.extend([transforms.Resize(256),
                                transforms.CenterCrop(224)])

        transf_list.extend([transforms.ToTensor(),
                            #utils.ToSpaceBGR(opt.input_space=='BGR'),
                            #utils.ToRange255(max(opt.input_range)==255),
                            normalize,
                            #transforms.Normalize(mean=opt.mean, std=opt.std)
                            # transforms.Lambda(lambda x: x[[2, 1, 0], ...])
                            ])
        self.transform = transforms.Compose(transf_list)

        # if self.is_validation or self.samples_per_class==1:
        self.image_list = [[(x,key) for x in self.image_dict[key]] for key in self.image_dict.keys()]
        self.image_list = [x for y in self.image_list for x in y]

        self.sample_probs = np.ones(len(self.image_list))/len(self.image_list)


    def ensure_3dim(self, img):
        if len(img.size)==2:
            img = img.convert('RGB')
        return img


    def __getitem__(self, idx):
        if not self.is_validation:
            add_cluster_info = ()
            if self.cluster_labels is not None:
                add_cluster_info = (self.cluster_labels[idx],)

            if self.samples_per_class==1:
                return add_cluster_info+(self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0]))))

            if self.n_samples_drawn==self.samples_per_class:
                #Once enough samples per class have been drawn, we choose another class to draw samples from.
                #Note that we ensure with self.classes_visited that no class is chosen if it had been chosen
                #previously or one before that.
                counter = copy.deepcopy(self.avail_classes)
                for prev_class in self.classes_visited:
                    if prev_class in counter: counter.remove(prev_class)
                self.current_class   = np.random.choice(counter)
                self.classes_visited = self.classes_visited[1:]+[self.current_class]
                self.n_samples_drawn = 0

            class_sample_idx = idx%len(self.image_dict[self.current_class])
            self.n_samples_drawn += 1

            return add_cluster_info+(self.current_class,self.transform(self.ensure_3dim(Image.open(self.image_dict[self.current_class][class_sample_idx]))))
        else:
            return (self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0]))))

    def __len__(self):
        return self.n_files





################## BASIC PYTORCH DATASET USED FOR ALL DATASETS ##################################
class ClusterDataset(Dataset):
    def __init__(self, image_paths, image_labels, opt):
        self.n_files     = len(image_paths)
        self.samples_per_class = opt.samples_per_class

        self.pars        = opt
        self.image_paths = image_paths
        self.update_labels(image_labels)
        ##### Option 2: Use Mean/Stds on which the networks were trained
        # transf_list = [transforms.RandomResizedCrop(size=224),transforms.ToTensor(),utils.ToSpaceBGR(opt.input_space=='BGR'),\
        #                utils.ToRange255(max(opt.input_range)==255),transforms.Normalize(mean=opt.mean, std=opt.std)]

        transf_list = [
            transforms.RandomResizedCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=opt.mean, std=opt.std)
        ]


        self.transform = transforms.Compose(transf_list)


    def update_labels(self, image_labels):
        self.avail_classes = np.unique(image_labels)
        self.indexer       = {i:np.where(image_labels==i)[0] for i in self.avail_classes}

        self.current_class   = np.random.randint(len(self.avail_classes))
        self.classes_visited = [self.current_class, self.current_class]
        self.n_samples_drawn = 0

        self.image_list = [[(self.image_paths[x],key) for x in self.indexer[key]] for key in self.indexer.keys()]
        self.image_list = [x for y in self.image_list for x in y]


    def ensure_3dim(self, img):
        if len(img.size)==2:
            img = img.convert('RGB')
        return img

    def __getitem__(self, idx):
        if self.samples_per_class==1:
            return (self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_paths[idx]))))

        if self.n_samples_drawn==self.samples_per_class:
            #Once enough samples per class have been drawn, we choose another class to draw samples from.
            #Note that we ensure with self.classes_visited that no class is chosen if it had been chosen
            #previously or one before that.
            counter = copy.deepcopy(list(self.avail_classes))
            for prev_class in self.classes_visited:
                if prev_class in counter: counter.remove(prev_class)
            self.current_class   = np.random.choice(counter)
            self.classes_visited = self.classes_visited[1:]+[self.current_class]
            self.n_samples_drawn = 0

        class_sample_idx = idx%len(self.indexer[self.current_class])
        self.n_samples_drawn += 1

        idx = self.indexer[self.current_class][class_sample_idx]
        return self.current_class,self.transform(self.ensure_3dim(Image.open(self.image_paths[idx])))


    def __len__(self):
        return self.n_files


class ClusterDataset_wLabels(Dataset):
    def __init__(self, image_paths, image_labels, gt_labels, opt):
        self.n_files     = len(image_paths)
        self.samples_per_class = opt.samples_per_class

        self.pars        = opt
        self.gt_labels = gt_labels
        self.image_paths = image_paths
        self.update_labels(image_labels)
        ##### Option 2: Use Mean/Stds on which the networks were trained
        # transf_list = [transforms.RandomResizedCrop(size=224),transforms.ToTensor(),utils.ToSpaceBGR(opt.input_space=='BGR'),\
        #                utils.ToRange255(max(opt.input_range)==255),transforms.Normalize(mean=opt.mean, std=opt.std)]

        transf_list = [
            transforms.RandomResizedCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=opt.mean, std=opt.std)
        ]


        self.transform = transforms.Compose(transf_list)


    def update_labels(self, image_labels):
        self.avail_classes = np.unique(image_labels)
        self.indexer       = {i:np.where(image_labels==i)[0] for i in self.avail_classes}

        self.current_class   = np.random.randint(len(self.avail_classes))
        self.classes_visited = [self.current_class, self.current_class]
        self.n_samples_drawn = 0

        self.image_list = [[(self.image_paths[x],key) for x in self.indexer[key]] for key in self.indexer.keys()]
        self.image_list = [x for y in self.image_list for x in y]


    def ensure_3dim(self, img):
        if len(img.size)==2:
            img = img.convert('RGB')
        return img

    def __getitem__(self, idx):
        if self.samples_per_class==1:
            return (self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_paths[idx]))))

        if self.n_samples_drawn==self.samples_per_class:
            #Once enough samples per class have been drawn, we choose another class to draw samples from.
            #Note that we ensure with self.classes_visited that no class is chosen if it had been chosen
            #previously or one before that.
            counter = copy.deepcopy(list(self.avail_classes))
            for prev_class in self.classes_visited:
                if prev_class in counter: counter.remove(prev_class)
            self.current_class   = np.random.choice(counter)
            self.classes_visited = self.classes_visited[1:]+[self.current_class]
            self.n_samples_drawn = 0

        class_sample_idx = idx%len(self.indexer[self.current_class])
        self.n_samples_drawn += 1

        idx = self.indexer[self.current_class][class_sample_idx]
        gt_label = self.gt_labels[idx]
        return self.current_class, gt_label, self.transform(self.ensure_3dim(Image.open(self.image_paths[idx])))


    def __len__(self):
        return self.n_files

# randomized triplets

class RandomizedDataset(Dataset):
    def __init__(self, image_paths, opt):
        self.n_files     = len(image_paths)
        self.samples_per_class = opt.samples_per_class

        self.pars        = opt
        self.image_paths = image_paths
        self.avail_classes = list(range(opt.num_cluster))

        ##### Option 2: Use Mean/Stds on which the networks were trained
        if opt.arch != 'googlenet':
            transf_list = [transforms.RandomResizedCrop(size=224),transforms.ToTensor(),utils.ToSpaceBGR(opt.input_space=='BGR'),\
                           utils.ToRange255(max(opt.input_range)==255),transforms.Normalize(mean=opt.mean, std=opt.std)]
        else:
            transf_list = [transforms.RandomResizedCrop(size=227),transforms.ToTensor(),utils.ToSpaceBGR(opt.input_space=='BGR'),\
                           utils.ToRange255(max(opt.input_range)==255),transforms.Normalize(mean=opt.mean, std=opt.std)]

        self.transform = transforms.Compose(transf_list)

        self.current_class   = random.choice(self.avail_classes)
        self.classes_visited = [self.current_class, self.current_class]
        self.n_samples_drawn = 0

    def ensure_3dim(self, img):
        if len(img.size)==2:
            img = img.convert('RGB')
        return img

    def __getitem__(self, idx):
        if self.n_samples_drawn==self.samples_per_class:
            #Once enough samples per class have been drawn, we choose another class to draw samples from.
            #Note that we ensure with self.classes_visited that no class is chosen if it had been chosen
            #previously or one before that.
            counter = copy.deepcopy(list(self.avail_classes))
            for prev_class in self.classes_visited:
                if prev_class in counter: counter.remove(prev_class)
            self.current_class   = np.random.choice(counter)
            self.classes_visited = self.classes_visited[1:]+[self.current_class]
            self.n_samples_drawn = 0

        self.n_samples_drawn += 1

        return self.current_class, self.transform(self.ensure_3dim(Image.open(self.image_paths[idx])))

    def update_num_classes(self, n_classes):
        self.avail_classes   = list(range(n_classes))
        self.current_class   = random.choice(self.avail_classes)
        self.classes_visited = [self.current_class, self.current_class]
        self.n_samples_drawn = 0

    def __len__(self):
        return self.n_files


class RandomizedDataset_wLabel(Dataset):
    def __init__(self, image_paths, gt_labels, opt):
        self.n_files     = len(image_paths)
        self.samples_per_class = opt.samples_per_class

        self.pars        = opt
        self.gt_labels = gt_labels
        self.image_paths = image_paths
        self.avail_classes = list(range(opt.num_cluster))

        ##### Option 2: Use Mean/Stds on which the networks were trained
        if opt.arch != 'googlenet':
            transf_list = [transforms.RandomResizedCrop(size=224),transforms.ToTensor(),utils.ToSpaceBGR(opt.input_space=='BGR'),\
                           utils.ToRange255(max(opt.input_range)==255),transforms.Normalize(mean=opt.mean, std=opt.std)]
        else:
            transf_list = [transforms.RandomResizedCrop(size=227),transforms.ToTensor(),utils.ToSpaceBGR(opt.input_space=='BGR'),\
                           utils.ToRange255(max(opt.input_range)==255),transforms.Normalize(mean=opt.mean, std=opt.std)]

        self.transform = transforms.Compose(transf_list)

        self.current_class   = random.choice(self.avail_classes)
        self.classes_visited = [self.current_class, self.current_class]
        self.n_samples_drawn = 0

    def ensure_3dim(self, img):
        if len(img.size)==2:
            img = img.convert('RGB')
        return img

    def __getitem__(self, idx):
        if self.n_samples_drawn==self.samples_per_class:
            #Once enough samples per class have been drawn, we choose another class to draw samples from.
            #Note that we ensure with self.classes_visited that no class is chosen if it had been chosen
            #previously or one before that.
            counter = copy.deepcopy(list(self.avail_classes))
            for prev_class in self.classes_visited:
                if prev_class in counter: counter.remove(prev_class)
            self.current_class   = np.random.choice(counter)
            self.classes_visited = self.classes_visited[1:]+[self.current_class]
            self.n_samples_drawn = 0

        self.n_samples_drawn += 1

        return self.current_class, self.gt_labels[idx], self.transform(self.ensure_3dim(Image.open(self.image_paths[idx])))

    def update_num_classes(self, n_classes):
        self.avail_classes   = list(range(n_classes))
        self.current_class   = random.choice(self.avail_classes)
        self.classes_visited = [self.current_class, self.current_class]
        self.n_samples_drawn = 0

    def __len__(self):
        return self.n_files


class PartlyRandomizedDataset_wLabel(Dataset):
    def __init__(self, image_paths, gt_labels, opt, random_prob=1.0):
        self.n_files     = len(image_paths)
        self.samples_per_class = opt.samples_per_class

        self.pars        = opt
        self.gt_labels = gt_labels

        self.image_paths = image_paths
        self.avail_classes = list(range(opt.num_cluster))

        self.gt_indexer = {i:np.where(gt_labels==i)[0] for i in self.avail_classes}
        self.random_prob = random_prob

        ##### Option 2: Use Mean/Stds on which the networks were trained
        if opt.arch != 'googlenet':
            transf_list = [transforms.RandomResizedCrop(size=224),transforms.ToTensor(),utils.ToSpaceBGR(opt.input_space=='BGR'),\
                           utils.ToRange255(max(opt.input_range)==255),transforms.Normalize(mean=opt.mean, std=opt.std)]
        else:
            transf_list = [transforms.RandomResizedCrop(size=227),transforms.ToTensor(),utils.ToSpaceBGR(opt.input_space=='BGR'),\
                           utils.ToRange255(max(opt.input_range)==255),transforms.Normalize(mean=opt.mean, std=opt.std)]

        self.transform = transforms.Compose(transf_list)

        self.current_class   = random.choice(self.avail_classes)
        self.classes_visited = [self.current_class, self.current_class]
        self.n_samples_drawn = 0

    def ensure_3dim(self, img):
        if len(img.size)==2:
            img = img.convert('RGB')
        return img

    def __getitem__(self, idx):
        if self.n_samples_drawn==self.samples_per_class:
            #Once enough samples per class have been drawn, we choose another class to draw samples from.
            #Note that we ensure with self.classes_visited that no class is chosen if it had been chosen
            #previously or one before that.
            counter = copy.deepcopy(list(self.avail_classes))
            for prev_class in self.classes_visited:
                if prev_class in counter: counter.remove(prev_class)
            self.current_class   = np.random.choice(counter)
            self.classes_visited = self.classes_visited[1:]+[self.current_class]
            self.n_samples_drawn = 0

        self.n_samples_drawn += 1

        # choose actual class sample by chance
        if np.random.rand(1) >= self.random_prob:
            idx = np.random.choice(self.gt_indexer[self.current_class], size=1)[0]

        return self.current_class, self.gt_labels[idx], self.transform(self.ensure_3dim(Image.open(self.image_paths[idx])))

    def update_num_classes(self, n_classes):
        self.avail_classes   = list(range(n_classes))
        self.current_class   = random.choice(self.avail_classes)
        self.classes_visited = [self.current_class, self.current_class]
        self.n_samples_drawn = 0

    def __len__(self):
        return self.n_files


# self-supervised

class RotationDataset(Dataset):
    def __init__(self, image_paths, opt):
        self.n_files     = len(image_paths)
        self.pars        = opt
        self.image_paths = image_paths

        ##### Option 2: Use Mean/Stds on which the networks were trained
        if opt.arch != 'googlenet':
            transf_list = [transforms.RandomResizedCrop(size=224),transforms.ToTensor(),utils.ToSpaceBGR(opt.input_space=='BGR'),\
                           utils.ToRange255(max(opt.input_range)==255),transforms.Normalize(mean=opt.mean, std=opt.std)]
        else:
            transf_list = [transforms.RandomResizedCrop(size=227),transforms.ToTensor(),utils.ToSpaceBGR(opt.input_space=='BGR'),\
                           utils.ToRange255(max(opt.input_range)==255),transforms.Normalize(mean=opt.mean, std=opt.std)]

        self.transform = transforms.Compose(transf_list)

    def ensure_3dim(self, img):
        if len(img.size)==2:
            img = img.convert('RGB')
        return img

    def __getitem__(self, idx):

        rots = [0, 90, 180, 270]

        # sample random rotation from {0, 90, 180, 270}
        class_id = np.random.choice([0, 1, 2, 3])
        rot_degree = rots[class_id]

        return class_id, self.transform(self.ensure_3dim(Image.open(self.image_paths[idx]).rotate(rot_degree)))

    def __len__(self):
        return self.n_files










# other

class ClusterSubsetDataset(Dataset):
    def __init__(self, image_list, image_paths, image_labels, opt):
        self.n_files     = len(image_paths)
        self.samples_per_class = opt.samples_per_class

        self.pars        = opt
        self.image_paths = image_paths
        self.update_labels(image_labels)
        ##### Option 2: Use Mean/Stds on which the networks were trained
        # transf_list = [transforms.RandomResizedCrop(size=224),transforms.ToTensor(),utils.ToSpaceBGR(opt.input_space=='BGR'),\
        #                utils.ToRange255(max(opt.input_range)==255),transforms.Normalize(mean=opt.mean, std=opt.std)]

        transf_list = [
            transforms.RandomResizedCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=opt.mean, std=opt.std)
        ]


        self.transform = transforms.Compose(transf_list)
        self.image_list = image_list


    def update_labels(self, image_labels):
        self.avail_classes = np.unique(image_labels)

        # remove 'unassigned class', i.e. lable = -1
        if -1 in self.avail_classes:
            self.avail_classes = self.avail_classes[1:]

        self.indexer       = {i:np.where(image_labels==i)[0] for i in self.avail_classes}

        self.current_class   = np.random.randint(len(self.avail_classes))
        self.classes_visited = [self.current_class, self.current_class]
        self.n_samples_drawn = 0


    def ensure_3dim(self, img):
        if len(img.size)==2:
            img = img.convert('RGB')
        return img

    def __getitem__(self, idx):
        if self.samples_per_class==1:
            return (self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0]))))

        if self.n_samples_drawn==self.samples_per_class:
            #Once enough samples per class have been drawn, we choose another class to draw samples from.
            #Note that we ensure with self.classes_visited that no class is chosen if it had been chosen
            #previously or one before that.
            counter = copy.deepcopy(list(self.avail_classes))
            for prev_class in self.classes_visited:
                if prev_class in counter: counter.remove(prev_class)
            self.current_class   = np.random.choice(counter)
            self.classes_visited = self.classes_visited[1:]+[self.current_class]
            self.n_samples_drawn = 0

        class_sample_idx = idx%len(self.indexer[self.current_class])
        self.n_samples_drawn += 1

        idx = self.indexer[self.current_class][class_sample_idx]
        return self.current_class,self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))


    def __len__(self):
        return self.n_files


class ClusterDataset_OLD(Dataset):
    def __init__(self, image_paths, image_labels, opt):
        self.n_files     = len(image_paths)
        self.samples_per_class = opt.samples_per_class

        self.pars        = opt
        self.image_paths = image_paths
        self.update_labels(image_labels)
        ##### Option 2: Use Mean/Stds on which the networks were trained
        # transf_list = [transforms.RandomResizedCrop(size=224),transforms.ToTensor(),utils.ToSpaceBGR(opt.input_space=='BGR'),\
        #                utils.ToRange255(max(opt.input_range)==255),transforms.Normalize(mean=opt.mean, std=opt.std)]

        transf_list = [
            transforms.RandomResizedCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=opt.mean, std=opt.std)
        ]


        self.transform = transforms.Compose(transf_list)


    def update_labels(self, image_labels):
        self.avail_classes = np.unique(image_labels)
        self.indexer       = {i:np.where(image_labels==i)[0] for i in self.avail_classes}

        self.current_class   = np.random.randint(len(self.avail_classes))
        self.classes_visited = [self.current_class, self.current_class]
        self.n_samples_drawn = 0

        self.image_list = [[(self.image_paths[x],key) for x in self.indexer[key]] for key in self.indexer.keys()]
        self.image_list = [x for y in self.image_list for x in y]


    def ensure_3dim(self, img):
        if len(img.size)==2:
            img = img.convert('RGB')
        return img

    def __getitem__(self, idx):
        if self.samples_per_class==1:
            return (self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0]))))

        if self.n_samples_drawn==self.samples_per_class:
            #Once enough samples per class have been drawn, we choose another class to draw samples from.
            #Note that we ensure with self.classes_visited that no class is chosen if it had been chosen
            #previously or one before that.
            counter = copy.deepcopy(list(self.avail_classes))
            for prev_class in self.classes_visited:
                if prev_class in counter: counter.remove(prev_class)
            self.current_class   = np.random.choice(counter)
            self.classes_visited = self.classes_visited[1:]+[self.current_class]
            self.n_samples_drawn = 0

        class_sample_idx = idx%len(self.indexer[self.current_class])
        self.n_samples_drawn += 1

        idx = self.indexer[self.current_class][class_sample_idx]
        return self.current_class,self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))


    def __len__(self):
        return self.n_files


class KNNDataset(Dataset):
    def __init__(self, image_list, knns, opt):
        self.image_paths = np.array([x[0] for x in image_list])
        self.image_classes = np.array([x[1] for x in image_list])
        self.n_files     = len(self.image_paths)
        self.samples_per_class = opt.samples_per_class

        self.pars        = opt
        self.update_knns(knns)
        ##### Option 2: Use Mean/Stds on which the networks were trained
        # transf_list = [transforms.RandomResizedCrop(size=224),transforms.ToTensor(),utils.ToSpaceBGR(opt.input_space=='BGR'),\
        #                utils.ToRange255(max(opt.input_range)==255),transforms.Normalize(mean=opt.mean, std=opt.std)]

        transf_list = [
            transforms.RandomResizedCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=opt.mean, std=opt.std)
        ]

        self.transform = transforms.Compose(transf_list)
        self.image_list = image_list


    def update_knns(self, knns):
        self.knns = knns
        self.current_class   = np.random.randint(len(self.knns))
        self.n_samples_drawn = 0


    def ensure_3dim(self, img):
        if len(img.size)==2:
            img = img.convert('RGB')
        return img


    def __getitem__(self, idx):
        # if self.samples_per_class==1:
        #     return (self.image_list[idx][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0]))))

        if self.n_samples_drawn == self.samples_per_class:
            #Once enough samples per class have been drawn, we choose another class to draw samples from.
            #Note that we ensure with self.classes_visited that no class is chosen if it had been chosen
            #previously or one before that.
            self.current_class   = idx
            self.n_samples_drawn = 1

            return self.current_class, self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))

        class_sample_idx = idx%len(self.knns[self.current_class])
        self.n_samples_drawn += 1

        idx = self.knns[self.current_class][class_sample_idx]
        return self.current_class, self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))


    def __len__(self):
        return self.n_files
