# -*- coding: utf-8 -*-
"""
file: json_dataset.py
date: 2020-08-14
"""


import os
import json
import random
import linecache
import torch




class JsonFileDataset(torch.utils.data.Dataset):
    def __init__(self, json_file_path, transform=None):
        """
        Args:
            json_file_path (string): Path to target json file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.json_file_path = json_file_path
        self.transform = transform

    def __len__(self):
        return int(os.popen("wc %s" % self.json_file_path).read().strip(" ").split(" ")[0])

    def __getitem__(self, idx):
        target_record = linecache.getline(self.json_file_path, idx + 1) 
        target_record = json.loads(target_record)

        if self.transform is None or self.transform == []:
            sample = target_record
        else:
            sample = self.transform(target_record)
        return sample



def build_json_files_dataset(data_path, epoch_num=1, if_shuffle=True, transform=None):
    datasets = []

    if os.path.isdir(data_path):
        files_path = [os.path.join(data_path, x) for x in os.listdir(data_path)]
    else:
        files_path = [data_path]

    for i in range(epoch_num):
        if if_shuffle:
            random.shuffle(files_path)
        #print(files_path)
        datasets = datasets + [JsonFileDataset(x, transform) for x in files_path] 

    datasets = torch.utils.data.ConcatDataset(datasets)
    return datasets


