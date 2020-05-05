#!/usr/bin/env python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

""" Prepare data for training sequence to sequence models,
the vocabulary is built from the dataset.
"""

import os
import sys
import re
import math
import random
import json
import cv2
import logging
import numpy as np
import argparse
from PIL import Image

import torch
from torch._utils import _accumulate
import torchvision.transforms as transforms
from torch.utils.data import Dataset, ConcatDataset, Subset
from data_loader.normalize_text import normalize_text
from data_loader.augment_data import get_augmenter
import seq2seq.training.tools as tools

LOGGER = logging.getLogger(__name__)

__author__ = "ThanhHoang"
__status__ = "Module"


class Sample(object):
    """ Sample from the dataset. """
    def __init__(self, file_path, label_text):
        """ Initialize the sample
        Arguments:
            file_path (str): path to the file
            label_text (str): label with respect to the path
        Return:
        """
        self.label_text = label_text
        self.file_path = file_path

class JsonHandler(object):
    """To read, dumpy json format. """
    def default(self, o):
        if isinstance(o, np.int64): return int(o)
        raise TypeError

    def read_json_file(self, filename):
        with open(filename, encoding='utf-8') as f:
            return json.load(f)

    def dump_to_file(self, data, filename):
        with open(filename, "w", encoding="utf-8") as fp:
            # json.dump(data, fp, default=self.default)
            json.dump(data, fp, indent=2, ensure_ascii=False)

class DataLoader(object):
    """ Data loader to loader batch samples for training process. """
    def __init__(self, opt, init_vocab=None):
        """ Modulate the data ratio in the batch.
        Arguments:
            opt (dict): predefined parameters.
        """
        LOGGER.debug("Initializing loading data set ............")
        train_data_collate = AlignCollate(\
            img_height=opt.img_height, img_width=opt.img_width, \
            keep_ratio_with_pad=opt.padding, add_augment=opt.augment, is_train=True)

        val_data_collate = AlignCollate(\
            img_height=opt.img_height, img_width=opt.img_width, \
            keep_ratio_with_pad=opt.padding, add_augment=False, is_train=False)
        self.batch_size = opt.batch_size

        if init_vocab != None:
            self.vocabulary = init_vocab
        else:
            self.vocabulary = set()

        multi_data_getter = MultiDataset()
        all_train, all_val = multi_data_getter.combine_datasets(opt)

        # Load train data 
        self.train_samples = GetDataset(samples=all_train, \
            charset=multi_data_getter.char_set, opt=opt)
        if init_vocab is None:
            self.vocabulary.update(self.train_samples.char_set)
        else:
            self.vocabulary = self.train_samples.update_vocab(self.vocabulary)
        LOGGER.debug("Number of training data: {0}".format(len(self.train_samples)))

        # Load validation data 
        self.val_samples = GetDataset(samples=all_val, \
            charset=multi_data_getter.char_set, opt=opt)
        if init_vocab is None:
            self.vocabulary.update(self.val_samples.char_set)
        else:
            self.vocabulary = self.val_samples.update_vocab(self.vocabulary)
        LOGGER.debug("Number of validation data: {0}".format(len(self.val_samples)))

        if init_vocab is None:
            # convert to list of characters
            self.vocabulary = list(self.vocabulary)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_samples, batch_size=self.batch_size,
            shuffle=True,
            num_workers=int(opt.workers),
            collate_fn=train_data_collate, pin_memory=True)

        self.val_loader = torch.utils.data.DataLoader(
            self.val_samples, batch_size=self.batch_size,
            shuffle=True,
            num_workers=int(opt.workers),
            collate_fn=val_data_collate, pin_memory=True)

class MultiDataset(object):
    """ Handle multi dataset - Combine all textlines data in data folders """
    def __init__(self):
        self.char_set = set()
        self.jsoner = JsonHandler()

    def process_sample(self, opt, root_path, item):
        # Change the file path and normalize label text
        filename, label = item
        filepath = os.path.join(root_path, filename)
        if opt.normalize_text == True:
            label_text = normalize_text(label)
        else:
            label_text = label
        
        # Add new character
        for char in label_text:
            if char not in self.char_set:
                self.char_set.add(char)
        return Sample(filepath, label_text)

    def combine_datasets(self, opt):
        all_train = list()
        all_val = list()

        # Get all the labeled lines in json files
        all_folder_paths = list(map(lambda f: \
                os.path.join(opt.data_path, f), \
                os.listdir(opt.data_path)))
        all_folder_paths = list(filter(os.path.isdir, all_folder_paths))

        # Get json label files in folders 
        all_train_dict = [self.jsoner.read_json_file(\
                os.path.join(f, opt.train_file)) for f in all_folder_paths]
        all_val_dict = [self.jsoner.read_json_file(\
                os.path.join(f, opt.val_file)) for f in all_folder_paths]

        # Normalize label text and add to train list
        for root_path, data_dict in zip(all_folder_paths, all_train_dict):
            LOGGER.debug("Getting train data from {}".format(root_path))
            all_train.extend(list(map(lambda item: \
                self.process_sample(opt, root_path, item), data_dict.items())))

        # Normalize label text and add to validation list
        for root_path, data_dict in zip(all_folder_paths, all_val_dict):
            LOGGER.debug("Getting val data from {}".format(root_path))
            all_val.extend(list(map(lambda item: \
                self.process_sample(opt, root_path, item), data_dict.items())))
        return (all_train, all_val)

class GetDataset(Dataset):
    """ Getting the data from given locations. """
    def __init__(self, samples, charset, opt):
        """ Initialize the GetDataset
        Arguments:
            label_file (str): the file containing the file path and its label
        """
        self.opt = opt
        self.char_set = charset
        self.samples = samples
        random.shuffle(self.samples)
        self.nSamples = len(self.samples)

    def update_vocab(self, char_list):
        """ Update characters from init character list with given data
        Args:
            char_list (lst): list of initialize character
        Return:
            char_list (lst): updated character list
        """
        for char in list(self.char_set):
            if char not in char_list:
                char_list.append(char)
        return char_list

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        label = self.samples[index].label_text
        img_path = self.samples[index].file_path
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # make dummy image and dummy label for corrupted image.
            if any(s < 1 for s in img.shape[:2]):
                img = np.zeros((self.opt.img_height, self.opt.img_width))
                label = '[dummy_label]'
        except IOError:
            LOGGER.debug("Corrupted image for ".format(index))

            # make dummy image and dummy label for corrupted image.
            img = np.zeros((self.opt.img_height, self.opt.img_width))
            label = '[dummy_label]'
        return (img, label)
    
class NormalizePAD(object):
    """ Normalize image with padding """
    def __init__(self, max_size, PAD_type='right', \
        is_augment=True, is_train=True):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type
        self.is_augment = is_augment
        self.is_train = is_train

    def __call__(self, img):
        if (self.is_augment == True \
            and self.is_train == True): 
            img = get_augmenter(img, probs=0.1)
        img = self.toTensor(img)
        # img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(1)
        Pad_img[:, :, :w] = img  # right pad
        # if self.max_size[2] != w:  # add border Pad
        #     Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(\
        #         2).expand(c, h, self.max_size[2] - w)
        Pad_img.sub_(0.5).div_(0.5) # mean 0.5, std 0.5
        return Pad_img

class AlignCollate(object):
    """ Preprocess images with resize and padding and normalize
    with mean & std resize by the max width image in the batch.
    """
    def __init__(self, img_height=64, img_width=1000, \
        keep_ratio_with_pad=True, add_augment=True, is_train=True):
        self.img_height = img_height
        self.max_img_width = img_width
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.add_augment = add_augment
        self.is_train = is_train

    def resize_img(self, img):
        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = math.ceil(self.img_height * ratio)
        if resized_w > self.max_img_width:
            resized_w = self.max_img_width
        resized_image = cv2.resize(img, \
            (resized_w, self.img_height), \
                interpolation=cv2.INTER_CUBIC)
        return (resized_image, resized_w)

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)
        outputs = list(map(lambda x: self.resize_img(x), images))

        # padding image by max width img in the batch
        images, w_values = map(list, zip(*outputs))
        transform = NormalizePAD((1, \
            self.img_height, max(w_values)), \
            is_augment=self.add_augment, is_train=self.is_train)

        if self.keep_ratio_with_pad:
            image_list = list(map(transform, images))
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_list], 0)
        else:
            image_list = list(map(transforms.ToTensor(), images))
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_list], 0)
        return (image_tensors, labels)


class InferAlignCollate(object):
    """ Preprocess images with resize and padding and normalize
    with mean & std resize by the max width image in the batch.
    """
    def __init__(self, img_height=64, img_width=1000, \
        keep_ratio_with_pad=True, add_augment=True, is_train=True):
        self.img_height = img_height
        self.max_img_width = img_width
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.add_augment = add_augment
        self.is_train = is_train

    def resize_img(self, img):
        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = math.ceil(self.img_height * ratio)
        if resized_w > self.max_img_width:
            resized_w = self.max_img_width
        resized_image = cv2.resize(img, \
            (resized_w, self.img_height), \
                interpolation=cv2.INTER_CUBIC)
        return (resized_image, resized_w)

    def __call__(self, list_imgs):
        list_imgs = filter(lambda x: x is not None, list_imgs)
        outputs = list(map(lambda x: self.resize_img(x), list_imgs))

        # padding image by max width img in the batch
        images, w_values = map(list, zip(*outputs))
        transform = NormalizePAD((1, \
            self.img_height, max(w_values)), \
            is_augment=self.add_augment, is_train=self.is_train)

        if self.keep_ratio_with_pad:
            image_list = list(map(transform, images))
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_list], 0)
        else:
            image_list = list(map(transforms.ToTensor(), images))
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_list], 0)
        return image_tensors

def get_predictor(self, type_pred, characters):
    dict_predictors = {
        "ctc": tools.CTCLabelConverter,
        "ace": tools.AggregationCrossEntropyLoss,
        "attn": tools.AttnLabelConverter
    }
    return dict_predictors[type_pred](characters)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """ Data params """
    parser.add_argument('--data_path', default='/opt/ml/input/data/train', help='path to dataset')
    
    
    parser.add_argument('--train_file', default='train.json', help='name of train label file')
    parser.add_argument('--val_file', default='val.json', help='name of train label val')
    parser.add_argument('--test_file', default='test.json', help='name of train label test')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)

    """ Training params """
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    """ Data processing """
    parser.add_argument('--img_height', type=int, default=64, help='the height of the input image')
    parser.add_argument('--img_width', type=int, default=1500, help='the width of the input image, before=800')
    parser.add_argument('--normalize_text', action='store_false', help='normalizing text label')
    parser.add_argument('--padding', action='store_false', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--augment', action='store_false', help='whether to augment data or not')

    """ Model Architecture """
    parser.add_argument('--prediction', type=str, default='ctc', help='Prediction stage. ctc|attn|ace')
    opt = parser.parse_known_args()[0]

    characters = None
    dataset_loader = DataLoader(opt, characters)
    train_loader = dataset_loader.train_loader
    val_loader = dataset_loader.val_loader

    """ set predictor """
    predictor = get_predictor(\
        opt.prediction, dataset_loader.vocabulary)
    vocabulary = predictor.character
    print('Number of new built characters: {0}'.format(len(self.vocabulary)))

