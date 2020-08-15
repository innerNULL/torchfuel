# -*- coding: utf-8 -*-
"""
file: demo4dataset.py
fate: 2020-08-14
"""


import sys
sys.path.append("../torchfuel")
import random
import json
import torch
import torchvision
from datasets import json_dataset
from datasets import json_transforms



def random_json_data_gen(target_path, sample_size):
    f = open(target_path, "w")

    for i in range(sample_size):
        curr_json_obj = {}

        group0_f0 = random.sample(range(100), 5)
        group1_f0 = random.sample(group0_f0, 1)
        curr_json_obj["group0_f0"] = group0_f0
        curr_json_obj["group1_f0"] = group1_f0

        curr_json_str = json.dumps(curr_json_obj, separators=(',', ':'))
        f.write(curr_json_str + "\n")

    f.close()
    return 0


def json_dataset_demo():
    demo_train_data_path = "./demo_data/"
    #demo_train_data_path = "./"
    demo_train_data_size = 100

    random_json_data_gen(demo_train_data_path + "part0", demo_train_data_size)
    random_json_data_gen(demo_train_data_path + "part1", int(demo_train_data_size * 1.2))
    random_json_data_gen(demo_train_data_path + "part2", int(demo_train_data_size * 0.8))
    part0_dataset = json_dataset.JsonFileDataset(demo_train_data_path + "part0", None)
    print(len(part0_dataset))
    for i in range(len(part0_dataset)): 
        print(part0_dataset[i])

    transforms = torchvision.transforms.Compose([json_transforms.StdJson2JsonTensor()]) 
    dataset = json_dataset.build_json_files_dataset(demo_train_data_path, 10, True, transforms)
    print(dataset, len(dataset))
    #for i in range(10):
    #    print(dataset[i])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    print(dataloader)
    for batch_num, batch_data in enumerate(dataloader):
        if batch_num > 20:
            break
        else:
            print(batch_data)



if __name__ == "__main__":
    json_dataset_demo()
