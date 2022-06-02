import math
import sys
import time
import torch
import cv2
import yaml 
import os
import numpy as np
import pandas as pd
from PIL import Image
import PIL
import torchvision.models.detection.mask_rcnn
from pathlib import Path

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils
import matplotlib.pyplot as plt
import torchvision.transforms.functional as fn

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
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        #print("during training 1" )
        #print(images[0])
        images = list(image.to(device) for image in images)
        #print("during training 2")
        #print(images[0])
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #print("during trianing 3" )
        #print(images)
        #print(targets)
        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        #torch.cuda.synchronize()
        model_time = time.time()
        #print("inside validation")
        #print("just before seding to the model image" )
        #print(images)
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

def load_classes():
    inv_map = {}
    for k, v in class_dict.items():
        inv_map[v] = k
    return inv_map

def myarea(box):
  return abs(box[2]-box[0])*abs(box[3]-box[1])

def intersectarea(box1, box2):
    
    xleft = max(int(box1[0]), int(box2[0]))
    xright = min(int(box1[2]), int(box2[2])) 
    ybottom = max(int(box1[1]), int(box2[1]))
    ytop = min(int(box1[3]), int(box2[3]))
    area = (ytop-ybottom)*(xright-xleft)
    if area<0:
      return 0
    else:  
      return area

class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']

        return {'img':((image.astype(np.float32)-self.mean)/self.std), 'annot': annots}

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

@torch.no_grad()
def return_boundingbox(model, data_loader, device):

    image_store_path = os.path.expanduser("./generated/images/")
    labels_store_path = os.path.expanduser("./generated/labels/")
    image_read_path = os.path.expanduser("/unlabeled/")
    labels_read_path = os.path.expanduser("/labelsstore/")  
    Path(image_store_path).mkdir(parents=True, exist_ok=True)
    Path(labels_store_path).mkdir(parents=True, exist_ok=True)
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.to(device)
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # coco = get_coco_api_from_dataset(data_loader.dataset)
    # iou_types = _get_iou_types(model)
    # coco_evaluator = CocoEvaluator(coco, iou_types)

    count = 0
    loopcount = 0
#     check = data_loader;
#     print("full metric logger print" )
#     print(check)
#     print("end of metric logge print")
    for images in metric_logger.log_every(data_loader, 100, header):
        loopcount = loopcount+1
        #print(images.shape)
        final_all_img = [img.to(device) for img in images]
        #final_all_img = images
        #outputs = model(final_all_img.to(device))
        outputs = model(final_all_img)
        #print(outputs)
        # count = count+1
        # if count==20:
        #     break
        #outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        unnormalize = UnNormalizer()
        for output in outputs:
          i = count
          count = count+1
          prediction = output
          boxes = prediction["boxes"].tolist()
          #boxes = convert_to_xywh(boxes).tolist()
          scores = prediction["scores"].tolist()
          classification = prediction["labels"].tolist()
          #get indices of final boxes and scores and labels needed
          threshold = 0.1
          #or directly get tuples of boxes and this here itself based on idx
          filtscoreidxs = [idx for idx, element in enumerate(scores) if element>threshold]
          if( len(filtscoreidxs) <= 0 ):
            continue
          filtboxes = [ boxes[u] for u in filtscoreidxs]
          filtscores = [ scores[u] for u in filtscoreidxs]
          filtclassification = [ classification[u] for u in filtscoreidxs]
          df = pd.DataFrame(list(zip( filtscoreidxs, filtboxes, filtscores, filtclassification )),
          columns =[ 'index', 'boxes', 'scores', "labels"])          
          df['area'] = df.apply(
      lambda row: abs(row['boxes'][2]-row['boxes'][0])*abs(row['boxes'][3]-row['boxes'][1]) , axis=1)
          df = df.sort_values(by=['area'])
          finalind = []
          addthis = 0
          fillen = len(filtscoreidxs)
          #print(fillen)
          first = 1
          for num in range(0,fillen):                            
          #for num, row in df.iterrows():
              #print("Enering with num ", num)
              row = df.iloc[num]
              #we first remove boxes which can be potentioal noise
              if(row['area'] <5000):
                continue
              if len(finalind)>=2:
                break
              if first==1:
                finalind.append(row['index'])
                first = 0
              else:  
                #print(num)
                currbox = row['boxes']
                addthis = 1
                for j in finalind: #the ones that we store are actual index
                  #currbox would be larger than previous one
                  areaint = intersectarea(currbox, filtboxes[j])
                  if (areaint/myarea(filtboxes[j])) > 0.7:
                    addthis=0
                    break
                if addthis:
                  finalind.append(row['index'])

          finboxes = [ filtboxes[z] for z in finalind]
          finscores = [ filtscores[z] for z in finalind]
          finclassification = [ load_classes()[filtclassification[z]] for z in finalind]

          im = Image.open( image_read_path + str(i)+ ".PNG");
          if(len(finboxes)<=0):
              continue
          mylen = len(finboxes)
          for j in range(mylen):
              im1 = im.crop(finboxes[j])
              im1 = fn.resize(im1, size=[224, 224])
              name = str(i) + "_" + str(j)
              im1 = im1.save(image_store_path + name + ".PNG" )
          yaml_data = {'labels': finclassification, 'scores': finscores }             
          yaml_path = labels_store_path + str(i) + ".yml"
          f = open(yaml_path, 'w')
          documents = yaml.dump(yaml_data, f)
          f.close()
        print("stored this batch", loopcount )
        #model_time = time.time() - model_time
        
    #print(all_outputs)

    return 1

