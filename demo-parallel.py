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
from pathlib import Path
from BYOL import BYOL
from SelfSupervisedModel import SelfSupervisedModel
import Resnet
import argparse
from torchvision.models.detection.backbone_utils import BackboneWithFPN
parser = argparse.ArgumentParser(description='Fine tuning')
parser.add_argument('--backbonetype', help='Type of Pretrained model')
parser.add_argument('--batchsize', type=int, help='batchsize')
parser.add_argument('--lr', help='Learning Rate')
parser.add_argument('--pretrained', help='Pretrained weights File')
parser.add_argument('--epochs', help='epochs')
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(num_classes, backbone=None):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    args.rank = 0
    args.dist_url = 'tcp://localhost:58472'
    args.world_size = args.ngpus_per_node
    args.batch_size=args.batchsize if args.batchsize != None else 8
    args.pretrained_type='barlow' if args.backbonetype == None else args.backbonetype
    args.lr=0.0005 if args.lr == None else args.lr
    args.epochs=30 if args.epochs == None else args.epochs
    args.weights = args.pretrained
    if  args.weights == None:
    	args.weights='barlowcrop20.pth' if args.pretrained_type == 'barlow' else 'SelfSupervisedModel_DistributedDataParallel_best.h5'
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)
    print("That's it!")
def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    pretrained = False
    num_classes = 101
    torch.cuda.empty_cache()
    if args.pretrained_type =='barlow':
        print('loading self-supervised weights from ', )
        backbone_dict = torch.load(args.weights, map_location='cuda:'+str(gpu))
    else:
        print('loading self-supervised weights from ', 'byol.pth')
        byol_state_dict=torch.load(args.weights, map_location='cuda:'+str(gpu))
        backbone_dict = {}
        for key in byol_state_dict:
            if 'online_network' in key:
                new_key = key.replace('module.online_network.', '')
                backbone_dict[new_key] = byol_state_dict[key]
    model = get_model(num_classes)
    model.backbone.body.load_state_dict(backbone_dict)
    model.cuda(gpu)
    model=torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    train_dataset = LabeledDataset(root='/labeled', split="training", transforms=get_transform(train=True))
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    per_device_batch_size = args.batch_size // args.world_size
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=per_device_batch_size, num_workers=2, pin_memory=True, collate_fn=utils.collate_fn, sampler=sampler)

    valid_dataset = LabeledDataset(root='/labeled', split="validation", transforms=get_transform(train=False))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True, collate_fn=utils.collate_fn)


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    num_epochs = args.epochs
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, gpu, epoch, print_freq=10, backbone=base, rank=args.rank)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, valid_loader, device=gpu)
        if args.rank == 0:
               torch.save(model.state_dict(), 'weights'+str(epoch)+'.pth')

    print("That's it!")
if __name__ == "__main__":
    main()
