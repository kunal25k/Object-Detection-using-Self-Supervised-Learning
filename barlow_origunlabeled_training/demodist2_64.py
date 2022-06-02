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


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass

def main():
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='Barlow Twins Training')
    # parser.add_argument('data', type=Path, metavar='DIR',
    #                     help='path to dataset')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loader workers')           
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
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
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)
    print("That's it!")

def main_worker(gpu, args):
    print("entering spawn on gpu ")
    print(gpu)
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = BarlowTwins(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
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

    print("gpu name")
    print(gpu)
    print("parsng dataset")
    #verify
    dataset = UnlabeledDataset(root='/unlabeled',transform= Transform())
    per_device_batch_size = 64
    args.workers = 2
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)

    start_time = time.time()
    args.epochs = 5
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        print("entering epochs distributed" )
        sampler.set_epoch(epoch)
        for step, (y1, y2) in enumerate(loader, start=epoch * len(loader)):   
        #for step, ((y1, y2), _) in enumerate(loader, start=epoch * len(loader)):
            print("entering just inside the loader")
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            print("entering the ost y gpu parrt")		
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
        if epoch % 10 == 0: 
            #store the backbone named model every 25 epochs - roughly 500 epochs we run, so 20 times saved -> 300*20 = 6 gb roughly
            print("storing logged resnet state now - which is stored every 10 epochs")
            # save final model
            torch.save(model.module.backbone.state_dict(),
                   args.checkpoint_dir / 'resnet50_' + str(epoch) + '.pth')
        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
    if args.rank == 0:
        # save final model
        torch.save(model.module.backbone.state_dict(),
                   args.checkpoint_dir / 'resnet50.pth')


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
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
