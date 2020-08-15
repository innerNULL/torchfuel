# -*- coding: utf-8 -*-
"""
file: json_transforms.py
date: 2020-08-15
"""


import torch


class StdJson2JsonTensor(object):
    def __call__(self, sample):
        out = {}

        for k, v in sample.items():
            if isinstance(v, list):
                if isinstance(v[0], int):
                    out[k] = torch.LongTensor(v)
                if isinstance(v[0], float):
                    out[k] = torch.FloatTensor(v)
            else:
                if isinstance(v, int):
                    out[k] = torch.LongTensor(v)
                if isinstance(v, float):
                    out[k] = torch.FloatTensor(v)
        #print(sample, out)
        return out
