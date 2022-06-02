# Feel free to modifiy this file. 
# It will only be used to verify the settings are correct 
# modified from https://pytorch.org/docs/stable/data.html

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import yaml 
import transforms as T
import torchvision.transforms.functional as fn
import utils
from engine import train_one_epoch, evaluate, return_boundingbox
from dataset import UnlabeledDataset, LabeledDataset
import sys
from pathlib import Path
from coco_eval import convert_to_xywh
import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse
import pandas as pd
from PIL import Image
import PIL  
# orig_stdout = sys.stdout
# f = open('out3.txt', 'w')
# sys.stdout = f
fullimgcount=0

class_dict = {
'cup or mug': 0,
'bird': 1,
'hat with a wide brim': 2,
'person': 3,
'dog': 4,
'lizard': 5,
'sheep': 6,
'wine bottle': 7,
'bowl': 8,
'airplane': 9,
'domestic cat': 10,
'car': 11,
'porcupine': 12,
'bear': 13,
'tape player': 14,
'ray': 15,
'laptop': 16,
'zebra': 17,
'computer keyboard': 18,
'pitcher': 19,
'artichoke': 20,
'tv or monitor': 21,
'table': 22,
'chair': 23,
'helmet': 24,
'traffic light': 25,
'red panda': 26,
'sunglasses': 27,
'lamp': 28,
'bicycle': 29,
'backpack': 30,
'mushroom': 31,
'fox': 32,
'otter': 33,
'guitar': 34,
'microphone': 35,
'strawberry': 36,
'stove': 37,
'violin': 38,
'bookshelf': 39,
'sofa': 40,
'bell pepper': 41,
'bagel': 42,
'lemon': 43,
'orange': 44,
'bench': 45,
'piano': 46,
'flower pot': 47,
'butterfly': 48,
'purse': 49,
'pomegranate': 50,
'train': 51,
'drum': 52,
'hippopotamus': 53,
'ski': 54,
'ladybug': 55,
'banana': 56,
'monkey': 57,
'bus': 58,
'miniskirt': 59,
'camel': 60,
'cream': 61,
'lobster': 62,
'seal': 63,
'horse': 64,
'cart': 65,
'elephant': 66,
'snake': 67,
'fig': 68,
'watercraft': 69,
'apple': 70,
'antelope': 71,
'cattle': 72,
'whale': 73,
'coffee maker': 74,
'baby bed': 75,
'frog': 76,
'bathing cap': 77,
'crutch': 78,
'koala bear': 79,
'tie': 80,
'dumbbell': 81,
'tiger': 82,
'dragonfly': 83,
'goldfish': 84,
'cucumber': 85,
'turtle': 86,
'harp': 87,
'jellyfish': 88,
'swine': 89,
'pretzel': 90,
'motorcycle': 91,
'beaker': 92,
'rabbit': 93,
'nail': 94,
'axe': 95,
'salt or pepper shaker': 96,
'croquet ball': 97,
'skunk': 98,
'starfish': 99
}


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_transform_unlabeled(train):
    transforms = []
    transforms.append(T.ToTensorUnlabeled())
    if train:
        #transforms.append(T.RandomHorizontalFlipJustImage(0.5))
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.ComposeJustImage(transforms)

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


store = 1
display = 0
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 101
# train_dataset = LabeledDataset(root='labeled/', split="training", transforms=get_transform(train=True), your_length = 10)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)

# valid_dataset = LabeledDataset(root='labeled/', split="validation", transforms=get_transform(train=False), your_length = 5)
# valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)

test_dataset = UnlabeledDataset(root='/unlabeled', transform=get_transform_unlabeled(train=False))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

model = get_model(num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

#model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
PATH = "./gcpfastrcnn_13.pth" 
#     PATH = "./gcpfastrcnn.pth" 
checkpoint = torch.load(PATH, map_location='cpu')
model.to(device)
model.load_state_dict(checkpoint['model'])
epoch = checkpoint['epoch']
  #   print(epoch)
  #  print(model)
#labels_store_path = os.path.expanduser("~/Desktop//masters /dl/semsslproject2/generated/labels/")
#labels_store_path = os.path.expanduser("labelsstore2/")
#Path(labels_store_path).mkdir(parents=True, exist_ok=True)

print("starting hre")
#evaluate(model, valid_loader, device=device)
success = return_boundingbox(model, test_loader, device= device)
print("is it success", success)


