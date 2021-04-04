import os
import numpy as np
import collections

import torch
torch.set_printoptions(profile="full")
import torch.nn as nn
from torch.autograd import Variable

from config import *


class Averager(object):
    """
    Compute average for `torch.Variable` and `torch.Tensor`. 
    """
    def __init__(self):
        self.reset()

    def add(self, loss):
        if isinstance(loss, Variable):
            count = loss.data.numel()
            loss = loss.data.sum()
        elif isinstance(loss, torch.Tensor):
            count = loss.numel()
            loss = loss.sum()

        self.n_count += count
        self.sum += loss

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def get_chars(split):
    annotations, _ = get_annotation(split)
    formulas_chars = []
    [[formulas_chars.append(f_char) for f_char in formula["formula_str"][0]] for formula in annotations]

    return list(set(formulas_chars))


def get_strs(split):
    annotations, _ = get_annotation(split)
    formulas_strs = []
    [formulas_strs.append(formula["formula_str"][0]) for formula in annotations]

    return formulas_strs


def get_map(formulas_chars):
    max_value = len(formulas_chars)
    chars2num = {}
    for i in range(max_value):
        chars2num[formulas_chars[i]] = str(i+3)
    # 0 is START, 1 is END, 2 is PADDING, so we should begin at 3
    return chars2num


def target_str_encode(formulas_strs, chars2num, START="0 ", END="1", PAD="2 "):
    print("Begin to get targets_tensors")

    l = []
    [l.append(len(formula_str)) for formula_str in formulas_strs]
    max_length = max(l)
    # print(max_length)

    targets_tensors = []
    for i in range(len(formulas_strs)):
        formula_strs = []
        formula_str = formulas_strs[i]
        # print(formula_str)

        for k in range(len(formula_str)):
            if k % 2 == 0:
                formula_strs.append(chars2num[formula_str[k]])
            else:
                formula_strs.append(" ")
        # print(formula_strs)

        formula_target = []
        formula_target.append(START)
        [formula_target.append(c) for c in formula_strs]
        formula_target.append(END)
        target_str = "".join(formula_target)
        target_list = target_str.split(" ")
        # print(len(target_list))

        target_tensor = torch.ones(max_length) * 2
        for i in range(len(target_list)):
            target_tensor[i] = int(target_list[i])
        # print(target_tensor)

        targets_tensors.append(target_tensor)
        # break
        # if you want to test de/encode, please open "break"

    print("Done!")

    return torch.tensor([item.numpy() for item in targets_tensors], dtype=torch.long)


def formula_str_decode(targets_tensors, chars2num):
    num2chars = {value:key for key, value in chars2num.items()}

    formulas_strs = []
    for i in range(len(targets_tensors)):
        target_strs = []
        target_tensor = targets_tensors[i]
        # print(target_str)

        for k in range(len(target_tensor)):
            if target_tensor[k] != 0 and target_tensor[k] != 1 and target_tensor[k] != 2:
                # print(type(target_tensor[k]))
                target_strs.append(num2chars[str(int(target_tensor[k]))])
            formula_str = "".join(target_strs)
        formulas_strs.append(formula_str)

    return formulas_strs


def get_data(split):
    print("Begin to get data of " + split)
    chars2num = get_map(get_chars(split))
    targets_tensors = target_str_encode(get_strs(split), chars2num)
    # formulas_strs = formula_str_decode(targets_tensors, chars2num)
    # print(formulas_strs)
    print("Data of " + split + ", Done!")

    return targets_tensors


def weights_init(model):
    # Official init from torch repo.
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    get_data("train")
    # you can test utils.py here
