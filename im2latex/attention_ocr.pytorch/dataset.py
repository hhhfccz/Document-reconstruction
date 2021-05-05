import six
import sys
import cv2
import numpy as np
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms

from utils import *


class Im2latex_Dataset(Dataset):

    def __init__(self, split, transform=None):
        self.split = split
        self.transform = transform

        annotations, _ = get_annotation(split)
        self.annotations = annotations
        self.length = len(annotations)

        labels, chars2num = get_data(split)
        self.labels = labels
        self.chars2num = chars2num

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        assert idx <= len(self), "index range error"

        formula_img_path = self.annotations[idx]["formula_road"]

        try:
            formula_img = cv2.imread(formula_img_path, 0)
        except IOError:
            print("Corrupted image for %d" % idx)

        if self.transform is not None:
            formula_img = self.transform(formula_img)

        formula_label = self.labels[idx]

        return (formula_img, formula_label)


class ResizeNormalize():

    def __init__(self, imgH, imgW):
        self.imgH = imgH
        self.imgW = imgW
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        # print(img.shape)
        img = cv2.resize(img, (self.imgW, self.imgH), interpolation=cv2.INTER_CUBIC)
        # print(img)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class RandomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                h, w = image.shape
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)
            # assure imgH >= imgW

        transform = ResizeNormalize(imgH, imgW)
        imgs = [transform(image) for image in images]
        imgs = torch.cat([t.unsqueeze(0) for t in imgs], 0)

        return imgs, labels


class DataPrefetcher:

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        img = self.img
        text = self.text
        if img is not None:
            img.record_stream(torch.cuda.current_stream())
            text.record_stream(torch.cuda.current_stream())
        self.preload()
        return img, text

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return

        with torch.cuda.stream(self.stream):
            self.img, self.texts = self.next_input
            self.text = torch.zeros(len(self.texts), len(self.texts[0]), dtype=torch.long)
            j = 0
            for txt in self.texts:
                self.text[j] = txt
                j += 1

            self.img = self.img.cuda(non_blocking=True)
            self.text = self.text.cuda(non_blocking=True)

