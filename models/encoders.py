###Models
import numpy as np
import pandas as pd
import timm
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torchvision.transforms as T

import faiss
import math

from transformers import AutoModel,AutoTokenizer, AutoModelForMaskedLM

import torch
import timm
from timm.data import resolve_data_config, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms_factory import create_transform
from torchvision import transforms as T

from sentence_transformers import SentenceTransformer 



###MLP
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048, num_layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


