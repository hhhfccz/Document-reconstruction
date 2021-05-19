# ——*——coding:utf-8——*——
# author: hhhfccz(胡珈魁) time:2021/3/18
import os
import random
import sys
import json
from collections import namedtuple, Counter

from numba import njit, prange

import numpy as np
import torch
import torch.nn as nn

torch.set_printoptions(profile="full")


def full2half(uchar):
    """DBC TO SBC"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 32
    elif 0xFF01 <= inside_code <= 0xFF5E:
        inside_code -= 0xfee0
    return chr(inside_code)


def get_annotation(split):
    # MODIFIED: use your own dir
    dataset_path = "/home/hhhfccz/im2latex/dataset/im2latex-100k/"
    annotation_path = dataset_path + "annotations_" + split + ".json"

    with open(annotation_path, "r", encoding="ISO-8859-1") as annotation_file:
        annotations = json.load(annotation_file)

    return annotations


def get_chars(split, annotations):
    formula_chars = []
    [[formula_chars.append(f_char) for f_char in formula["formula_str"][0]] for formula in annotations]
    ans = list(set(formula_chars))
    return ans


def get_strs(split, annotations):
    formulas_strs = []
    [formulas_strs.append(formula["formula_str"]) for formula in annotations]
    return formulas_strs


def get_map(formulas_chars):
    max_value = len(formulas_chars)
    chars2num = {}
    for i in range(max_value):
        chars2num[formulas_chars[i]] = str(i + 3)
    # chars2num["¡"] = str(max_value + 3)
    # ¡ is a special char in test data
    # 0 is START, 1 is END, 2 is PADDING, so we should begin at 3
    return chars2num


@njit(parallel=True)
def get_targets_tensors(strs_length, max_length, formulas_strs, num2chars, target, START=0, END=1, PAD=2):
    length_num_chars = len(num2chars)
    for i in prange(strs_length):
        formula_target = np.ones(max_length+2) * 2
        formula_target[0] = START
        p = 1

        formula_str = formulas_strs[i]
        length_formula_str = len(formula_str)

        for k in prange(length_formula_str):
            for l in prange(length_num_chars):
                if num2chars[l] == formula_str[k]:
                    formula_target[p] = l + 3
                    p += 1

        formula_target[p] = END
        target[i, :] = formula_target
    return target


def get_namedtuple(chars2num):
    keys_list = list(chars2num.keys())
    values_list = list(chars2num.values())

    # namedtuple don't support some special characters and numbers, so i'll use ASCII code instead
    values_list_ascii = []
    nums_list = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    # "a" is not in nums_list, so we can use "a" as the hyphen
    [values_list_ascii.append("a".join([nums_list[int(n)] for n in str(values)])) for values in values_list]
    # print(values_list_ascii)

    Decode_chars = namedtuple("Decode_chars", values_list_ascii)
    num2chars_namedtuple = Decode_chars._make(keys_list)

    num2chars_namedtuple_json = json.dumps(num2chars_namedtuple._asdict(), indent=4, ensure_ascii=True)
    with open("num_chars_nametuple.json", "w", encoding="ISO-8859-1") as json_file:
        json.dump(num2chars_namedtuple_json, json_file, ensure_ascii=True)

    return num2chars_namedtuple


def target_str_encode(split, formulas_strs, chars2num):
    # print(formulas_strs)
    print("Begin to get targets_tensors")
    if split == "train":
        num2chars_namedtuple = get_namedtuple(chars2num)
    else:
        with open("num_chars_nametuple.json", "r", encoding="ISO-8859-1") as json_file:
            num2chars_json = json.load(json_file)
            num2chars = json.loads(num2chars_json)
        keys_list = list(num2chars.keys())
        values_list = list(num2chars.values())
        Decode_chars = namedtuple("Decode_chars", keys_list)
        num2chars_namedtuple = Decode_chars._make(values_list)
    print(chars2num)
    print(num2chars_namedtuple)
    # the name of chars is the English of ASCII code of nums

    l = []
    [l.append(len(formula_str)) for formula_str in formulas_strs]
    max_length = max(l)
    strs_length = len(formulas_strs)

    targets = np.zeros((strs_length, max_length+2))
    targets_tensors = get_targets_tensors(strs_length, max_length, formulas_strs, num2chars_namedtuple, targets)
    # print(targets_tensors)

    print("Done!")

    return torch.from_numpy(targets_tensors).long()


def formula_str_decode(targets_tensors, chars2num):
    num2chars = {value: key for key, value in chars2num.items()}

    formulas_strs = []
    for i in range(len(targets_tensors)):
        target_strs = []
        target_tensor = targets_tensors[i]

        for k in range(len(target_tensor)):
            if target_tensor[k] != 0 and target_tensor[k] != 1 and target_tensor[k] != 2:
                target_strs.append(num2chars[str(int(target_tensor[k]))])
            formula_str = "".join(target_strs)
            formulas_strs.append(formula_str)

    return formulas_strs


def get_data(split, annotations):
    """
    chars2num of test an valid are the subsets of chars2num of train
    """
    print("Begin to get data of " + split)
    if split == "train":
        chars2num = get_map(get_chars(split, annotations))
        chars2num_json = json.dumps(chars2num, indent=4, sort_keys=True, ensure_ascii=True)
        with open("encode-decode.json", "w", encoding="ISO-8859-1") as json_file:
            json.dump(chars2num_json, json_file, ensure_ascii=True)
    else:
        with open("encode-decode.json", "r", encoding="ISO-8859-1") as json_file:
            chars2num_json = json.load(json_file)
            chars2num = json.loads(chars2num_json)
    targets_tensors = target_str_encode(split, get_strs(split, annotations), chars2num)
    # print(targets_tensors)

    # targets_tensors = torch.tensor(targets_tensors_numpy, dtype=torch.long)
    print("Data of " + split + ", Done!")

    if split == "train":
        print("the correspondence between tex chars and numbers: \n", chars2num_json)
        return targets_tensors, chars2num
    else:
        return targets_tensors


def weights_init(model):
    """
    Official init from torch repo.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, random.random())
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)


def enlarge_decoder_output(x, max_enlargement=5):
    """
    enlarge the loss, use the derivative in tanh
    """
    x /= 1e3
    return (pow(2, max_enlargement) - 1) * (1 - pow((1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x)), 2)) + 1


def get_decoder_input(texts, i):
    decoder_input = torch.LongTensor(len(texts))
    k = 0
    for text in texts:
        decoder_input[k] = text[i]
        k += 1
    return decoder_input


@njit(parallel=True)
def get_acc(length, decoded_labels, text):
    """
    get the acc, use numba to speed up
    """
    num_total = 0
    num_correct = 0
    for k in prange(length):
        for i in prange(len(decoded_labels[k])):
            pred = int(decoded_labels[k][i])
            target = int(text[:, k][i])
            if target == 0:
                continue
            elif target != 1:
                num_total += 1
                if pred == target:
                    num_correct += 1
            else:
                break
    # print(num_correct, num_total)
    acc = num_correct / float(num_total)
    return acc


if __name__ == "__main__":
    pass
    # you can test img_preprocess.py here
