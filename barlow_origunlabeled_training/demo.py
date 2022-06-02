# Feel free to modifiy this file. 
# It will only be used to verify the settings are correct 
# modified from https://pytorch.org/docs/stable/data.html

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import transforms as T
from barlow import BarlowTwins, LARS, Transform
import utils
from engine import train_one_epoch, evaluate
from dataset import UnlabeledDataset, LabeledDataset

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

#def get_transform_barlow(train):


# def handle_sigusr1(signum, frame):
#     os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
#     exit()


# def handle_sigterm(signum, frame):
#     pass


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parser = argparse.ArgumentParser(description='Barlow Twins Training')
    
    #parser.add_argument('data', type=Path, metavar='DIR',
    #                    help='path to dataset')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loader workers')
    #can change epochs here                  - defualt was 1000
    parser.add_argument('--epochs', default=4, type=int, metavar='N',
                        help='number of total epochs to run')
    #batch_size = 256, default was 2048
    parser.add_argument('--batch-size', default=2048, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                        help='base learning rate for weights')
    parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                        help='base learning rate for biases and batch norm parameters')
    parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                        help='weight decay')
    parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                        help='weight on off-diagonal terms')
    parser.add_argument('--projector', default='8192-8192-8192', type=str,
                        metavar='MLP', help='projector MLP')
    parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                        help='print frequency')
    parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                        metavar='DIR', help='path to checkpoint directory')
    torch.cuda.empty_cache()
    print(torch.cuda.device_count())
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    # if 'SLURM_JOB_ID' in os.environ:
    #     # single-node and multi-node distributed training on SLURM cluster
    #     # requeue job on SLURM preemption
    #     signal.signal(signal.SIGUSR1, handle_sigusr1)
    #     signal.signal(signal.SIGTERM, handle_sigterm)
    #     # find a common host name on all nodes
    #     # assume scontrol returns hosts in the same order on all nodes
    #     cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
    #     stdout = subprocess.check_output(cmd.split())
    #     host_name = stdout.decode().splitlines()[0]
    #     args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
    #     args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
    #     args.dist_url = f'tcp://{host_name}:58472'
    # else:
        # single-node distributed training
    args.rank = 0
    args.dist_url = 'tcp://localhost:58472'
    args.world_size = args.ngpus_per_node
    print("entering before spwan" )
    main_worker(args.ngpus_per_node, args)
    #torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)
    #verify
    num_classes = 101

    # train_dataset = LabeledDataset(root='/labeled', split="training", transforms=get_transform(train=True))
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)
    # #model = get_model(num_classes)
    # model.to(device)

    # params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # num_epochs = 1

    # for epoch in range(num_epochs):
    #     # train for one epoch, printing every 10 iterations
    #     train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
    #     # update the learning rate
    #     lr_scheduler.step()
    #     # evaluate on the test dataset
    #     evaluate(model, valid_loader, device=device)

    print("That's it!")


def main_worker(gpu, args):
    print(gpu)
    #args.rank += gpu
    args.rank = 0 #we are just using it for printing
    # torch.distributed.init_process_group(
    #      backend='nccl', init_method=args.dist_url,
    #      world_size=args.world_size, rank=args.rank)

    # if args.rank == 0:
    print("entering spawn")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
    print(' '.join(sys.argv))
    print(' '.join(sys.argv), file=stats_file)
    #print(' '.join(sys.argv), stats_file)
    torch.cuda.empty_cache()
    # torch.cuda.set_device(gpu)
    # torch.backends.cudnn.benchmark = True
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    model = BarlowTwins(args)
    model.to(device)
    print("came till here")
    #verify
    #nmodel = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #nmodel = nn.SyncBatchNorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None)
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0
    print("parsng datasdet")
    print(gpu)
    #verify
    dataset = UnlabeledDataset(root='/unlabeled',transform= Transform())
    #loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)
    #dataset = torchvision.datasets.ImageFolder(args.data / 'train', Transform())        
    #sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    #assert args.batch_size % args.world_size == 0
    #per_device_batch_size = args.batch_size // args.world_size
    per_device_batch_size = 128
#     loader = torch.utils.data.DataLoader(
#         dataset, batch_size=per_device_batch_size, num_workers=args.workers,
#         pin_memory=True, sampler=sampler)
    args.workers = 2
    loader = torch.utils.data.DataLoader(
        #dataset, batch_size=per_device_batch_size, num_workers=args.workers,collate_fn=utils.collate_fn,
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True)


    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    args.epochs = 500
    for epoch in range(start_epoch, args.epochs):
        #sampler.set_epoch(epoch)
        print("entering epichs" )
        #train_one_epoch(
        ##for step, ((y1, y2), _) in enumerate(loader, start=epoch * len(loader)):

        # for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # images = list(image.to(device) for image in images)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # loss_dict = model(images, targets)

        # losses = sum(loss for loss in loss_dict.values())

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        # loss_value = losses_reduced.item()

        #for step, ((y1, y2), _) in enumerate(loader, start=epoch * len(loader)):
        for step, (y1, y2) in enumerate(loader, start=epoch * len(loader)):   
            y1 = y1.cuda(0)
            y2 = y2.cuda(0)
            #print(device)
            #y1.to(device)
            #y2.to(device)            
            #step = torch.FloatTensor(2)
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2)
            #loss = model.forward(y1,y2)    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()
            
            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
                    #print(json.dumps(stats), stats_file)
        if args.rank == 0:
            # save checkpoint
            #model = torch.nn.DataParallel(model) 
            print("storing now")
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
    if args.rank == 0:
        # save final model
        #model = torch.nn.DataParallel(model) 
        #torch.save(model.module.backbone.state_dict(),
        torch.save(model.backbone.state_dict(),
                   args.checkpoint_dir / 'resnet50.pth')

def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    #step = torch.FloatTensor(0)
    if step < warmup_steps:
     lr = base_lr * step / warmup_steps
    else:
         step -= warmup_steps
         max_steps -= warmup_steps
         q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
         end_lr = base_lr * 0.001
         lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases

if __name__ == "__main__":
    main()
