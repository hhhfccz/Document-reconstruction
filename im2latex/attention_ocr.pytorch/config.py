import os
import json

import cv2

import torch
from torch.utils.data import dataloader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


def get_device():
	if torch.cuda.is_available():
		return torch.device("cuda")
	else:
		return torch.device("cpu")


def to_device(data, device):
	if isinstance(data, (list, tuple)):
		return [to_device(x, device) for x in data]
	return data.to(device, non_blocking=True)


def get_annotation(split):
	# use your own dir
	data_dir = "/home/hhhfccz/im2latex/dataset/" + split
	annotation_path = "/home/hhfccz/im2latex/dataset/annotations_" + split + ".json"

	with open(annotation_path, "r", encoding="ISO-8859-1") as annotation_file:
		annotations = json.load(annotation_file)

	return annotations, data_dir
