# Feel free to modifiy this file. 
# It will only be used to verify the settings are correct 
# modified from https://pytorch.org/docs/stable/data.html

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import transforms as T
import utils
from engine import train_one_epoch, evaluate


from dataset import UnlabeledDataset, LabeledDataset

from BYOL import BYOL
from SelfSupervisedModel import SelfSupervisedModel
import Resnet

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.Resize((225, 225)))
    return T.Compose(transforms)

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    pretrained = False
    base = Resnet.Resnet50(pretrained=pretrained)
    byol = BYOL(base)
    ss_model = SelfSupervisedModel(byol, None, None)
    print('loading self-supervised weights from ', ss_model.weights_file_name)
    byol_state_dict = torch.load('resnet50_30.pth', map_location='cuda:0')
#     backbone_dict = {}
#     for key in byol_state_dict:
#         if 'online_network' in key:
#             new_key = key.replace('module.online_network.', '')
#             backbone_dict[new_key] = byol_state_dict[key]
    base.load_state_dict(byol_state_dict)
    base.to(device)
    num_classes = 101
    train_dataset = LabeledDataset(root='/scratch/ar6381/DL/demo/labeled/labeled', split="training", transforms=get_transform(train=True))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True, collate_fn=utils.collate_fn)

    valid_dataset = LabeledDataset(root='/scratch/ar6381/DL/demo/labeled/labeled', split="validation", transforms=get_transform(train=False))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True, collate_fn=utils.collate_fn)

    model = get_model(num_classes)
    model.to(device)
#     model=torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
#     model.cuda(gpu)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    num_epochs = 5

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10, backbone=base)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, valid_loader, device=device)

    print("That's it!")

if __name__ == "__main__":
    main()
