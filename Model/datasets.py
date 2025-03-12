import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from PIL import Image

import random
from torchvision.transforms import ToTensor
class BatchImagePairLoader:
    def __init__(self, data_dir, label_dir, height, width, batch_size=32, shuffle=True, target_channel = 3):
        self.target_channel = target_channel
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.height = height
        self.width = width

        self.data_files = [os.path.join(self.data_dir, f) for f in sorted(os.listdir(data_dir)) if f.endswith('.jpg')]
        self.label_files = [os.path.join(self.label_dir, f) for f in sorted(os.listdir(label_dir)) if f.endswith('.jpg')]
        self.indexes = list(range(len(self.data_files)))

        if self.shuffle:
            random.shuffle(self.indexes)

        #Define normalization transform
        #Define normalization transform
        
        # self.transform = transforms.Compose(
        #     [
        #         transforms.Resize(height),
        #         transforms.CenterCrop(height),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        #     ]
        # )
        self.transform = ToTensor()

    def __len__(self):
        return len(self.data_files) // self.batch_size

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data_files = [self.data_files[i] for i in batch_indexes]
        batch_label_files = [self.label_files[i] for i in batch_indexes]

        batch_data = []
        batch_labels = []
        batch_filenames = []

        for data_file, label_file in zip(batch_data_files, batch_label_files):
            data = Image.open(data_file).convert('RGB')
            label = Image.open(label_file).convert('RGB')

            data = data.resize((self.width, self.height))
            label = label.resize((self.width, self.height))

            data = self.transform(data).cuda()
            label = self.transform(label).cuda()

            batch_data.append(data)
            batch_labels.append(label)
            batch_filenames.append((os.path.basename(data_file), os.path.basename(label_file)))

        batch_data = torch.stack(batch_data)#.to(torch.float64)
        batch_labels = torch.stack(batch_labels)#.to(torch.float64)
        return batch_data, batch_labels
