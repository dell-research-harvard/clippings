###Download script

####Inference CLIP

####Continue pretrain clip
###This script is to continue pretraining clip on the japanese dataset. Our dataset is a df of image-text pairs

import pandas as pd
import numpy as np
import wandb
from utils.datasets_utils import *

import faiss 
from tqdm import tqdm

import math
import json
import argparse

from sklearn.model_selection import train_test_split


from PIL import Image
import torch


from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import InferenceModel, FaissKNN
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, CosineAnnealingLR, StepLR

from torch import nn
 
import wandb

from utils.datasets_utils import *


import data_loaders
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel
import encoders
import networkx as nx

##NX community detection
import networkx.algorithms.community as nx_comm
##Import combinations
from itertools import combinations
import subprocess

clip_transform=CLIP_BASE_TRANSFORM_CENTER


def transformed_image(image,transformer):
    image_trans = transformer(image)
    ##add batch dimension
    return image_trans


###Run as scrpt
if __name__ == "__main__":
    ###Load image
    image_path = "/mnt/data02/captions/test_day_pulled_crops/3948_121_mini_batch_4295_na_gap_1968.png"
    image = Image.open(image_path)

    ###Transform image
    transformed_image = clip_transform(image)

    ###Rclone to dropbox
    subprocess.run(["rclone", "copy", image_path, "maindb:abhishek_for_rclone/eval_clippings/"])




