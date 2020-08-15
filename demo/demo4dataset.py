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
from datasets import json_dataset




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

    dataset = json_dataset.build_json_files_dataset(demo_train_data_path, 10, True, None)
    print(dataset, len(dataset))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    print(dataloader)



if __name__ == "__main__":
    json_dataset_demo()
