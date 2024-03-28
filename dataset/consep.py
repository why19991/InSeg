import os
import random
import copy

import numpy as np
import torch
import torch.utils.data as data
import torchvision as tv
from PIL import Image
from torch import distributed

from dataset import transform
from .utils import Subset, filter_images, group_images

classes = {
    0: 'background',
    1: 'class 1',
    2: 'class 2',
    3: 'class 3',
    4: 'class 4'
}

class CoNSePSegmentation(data.Dataset):
    """
    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        is_aug (bool, optional): If you want to use the augmented train set or not (default is True)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, root, image_set='train', is_aug=True, transform=None):

        self.root = os.path.expanduser(root)
        self.year = 2020

        self.transform = transform

        self.image_set = image_set
        monusac_root = self.root
        splits_dir = os.path.join(monusac_root,'splits')

        if not os.path.exists(monusac_root):
            raise RuntimeError(
                f'Dataset {monusac_root} not found or corrupted.'
            )

        split_f = os.path.join(splits_dir,image_set.rstrip('\n') + '.txt')
        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val" '
                f'{split_f}'
            )

        # remove leading \n
        with open(os.path.join(split_f), "r") as f:
            file_names = [x[:-1].split(" ") for x in f.readlines()]


        self.images = [
            (
                os.path.join(monusac_root,x[0][1:]), os.path.join(monusac_root, x[1][1:])
             ) for x in file_names
        ]

    def __getitem__(self, index):
        """
           Args:
               index (int): Index
           Returns:
               tuple: (image, target) where target is the image segmentation.
           """

        img = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])
        # img = Image.open("/data/wzz/MoNuSAC/train/image_crop/TCGA-5P-A9K0-01Z-00-DX1_1_11.png").convert('RGB')
        # target = Image.open("/data/wzz/MoNuSAC/train/mask_crop/TCGA-5P-A9K0-01Z-00-DX1_1_11.png")
        if self.transform is not None:
            img, target = self.transform(img, target)

        # return img, target, "TCGA-5P-A9K0-01Z-00-DX1_1_11"
        return img, target, self.images[index][1][-31:-4]


    def __len__(self):
        return len(self.images)

    def viz_getter(self, index):
        image_path = self.images[index][0]
        raw_image = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])
        if self.transform is not None:
            img, target = self.transform(raw_image, target)
        else:
            img = copy.deepcopy(raw_image)
        return image_path, raw_image, img, target


class CoNSePSegmentationIncremental(data.Dataset):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 labels=None,
                 labels_old=None,
                 idxs_path=None,
                 masking=True,
                 overlap=True,
                 data_masking="current",
                 test_on_val=False,
                 **kwargs
                 ):

        full_consep = CoNSePSegmentation(root,'train' if train else 'val',is_aug= True,transform=None)
        self.labels = []
        self.labels_old = []

        if labels is not None:

            labels_old = labels_old if labels_old is not None else []

            self.__strip_zero(labels)
            self.__strip_zero(labels_old)

            assert not any(
                l in labels_old for l in labels
            ), "labels and labels_old must be disjoint sets"

            self.labels = [0] + labels
            self.labels_old = [0] + labels_old

            self.order = [0] + labels_old + labels

            if idxs_path is not None and os.path.exists(idxs_path):
                idxs = np.load(idxs_path).tolist()

            else:
                idxs = filter_images(full_consep, labels, labels_old,overlap=overlap)
                if idxs_path is not None and distributed.get_rank() == 0:
                    np.save(idxs_path,np.array(idxs,dtype=int))

            if test_on_val:
                rnd = np.random.RandomState(1)
                rnd.shuffle(idxs)
                train_len = int(0.8 * len(idxs))
                if train:
                    idxs = idxs[:train_len]
                else:
                    idxs = idxs[train_len:]

            masking_value = 0   # Future classes will be considered as background.
            self.inverted_order = {label: self.order.index(label) for label in self.order}
            self.inverted_order[255] = 255
            reorder_transform = tv.transforms.Lambda(
                lambda t: t.apply_(
                    lambda x: self.inverted_order[x] if x in self.inverted_order else masking_value
                )
            )

            if masking:
                if data_masking == 'current':
                    tmp_labels = self.labels + [255]
                elif data_masking == 'current+old':
                    tmp_labels = labels_old + self.labels + [255]
                elif data_masking == 'all':
                    raise NotImplementedError(
                        f"data_masking={data_masking} not yet implemented sorry not sorry."
                    )

                elif data_masking == "new":
                    tmp_labels = self.labels
                    masking_value = 255

                    target_transform = tv.transforms.Lambda(
                        lambda t: t.
                            apply_(lambda x: self.inverted_order[x] if x in tmp_labels else masking_value)
                    )
                else:
                    assert False
                target_transform = reorder_transform

                # make the subset of the dataset
                self.dataset = Subset(full_consep, idxs, transform, target_transform)
            else:
                self.dataset =  full_consep



    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        return self.dataset[index]