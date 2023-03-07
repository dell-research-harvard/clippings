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



def get_image_text_embeddings(data_loader,clip_model,mlp_model,device,processor,pooling_type,im_wt):
    clip_model.eval()
    if not mlp_model is None:
        mlp_model.eval()

    for batch_idx, (text, image_data, labels, image_path) in tqdm(enumerate(data_loader)):

        labels = labels.to(device)

        ####Unsquueze the image data
        image_data = image_data.to(device)

        ### text is a tuple of strings, we need to convert it to a tensor
        text=list(text)
        text_features = processor.tokenizer(text, return_tensors="pt", padding=True,max_length=77,truncation=True)

        for key in text_features.keys():
            text_features[key]=text_features[key].to(device)
        

        with torch.no_grad():


            model_output=clip_model.forward(input_ids=text_features["input_ids"],pixel_values=image_data,
                                                attention_mask=text_features["attention_mask"])
            image_embeds, text_embeds = model_output["image_embeds"], model_output["text_embeds"]

            # final_embeds=torch.cat((image_embeds,text_embeds),dim=1)
            ###MEan of the two embeddings
            if pooling_type=="mean":
                final_embeds= im_wt*image_embeds + (1-im_wt)*text_embeds
            elif pooling_type=="mlp":
                ###Use an MLP to combine the image and text embeddings
                ###concat the image and text embeddings
                final_embeds=torch.cat([image_embeds,text_embeds],dim=1)
                ###Pass through an MLP
                final_embeds=mlp_model.forward(final_embeds)
                
            # final_embeds=text_embeds
            final_embeds=final_embeds/torch.norm(final_embeds, dim=1, keepdim=True)

            ####
            if batch_idx == 0:
                all_embeddings = final_embeds
                all_labels = labels
                all_text=text
                all_paths=image_path
            else:
                all_embeddings = torch.cat((all_embeddings, final_embeds), dim=0)
                all_labels = torch.cat((all_labels, labels), dim=0)
                all_text=all_text+text
                all_paths=all_paths+image_path

    return all_embeddings, all_labels, all_text, all_paths


###Run as script
if __name__ == "__main__":

    checkpoint_path="/mnt/data01/clippings_general/models/bienc_clip_pretrain_labelled_m3.pt"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    clip_transform=CLIP_BASE_TRANSFORM_CENTER
    ###Load checkpoint
    if checkpoint_path is not None:
        clip_model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))
    # model.load_state_dict(torch.load("/mnt/data01/clippings_general/models/bienc_clip_pretrain_labelled_m3.pt", map_location=torch.device(device)))
    clip_model.to(device)

    ###Load data
    eval_data=pd.read_csv("/mnt/data01/clippings_general/texts/labelled_news_eval_reformatted.csv")

    ##Load the dataset
    ###Create the data datsets
    eval_dataset=data_loaders.TextImageDataset(eval_data, img_transform=clip_transform)
    eval_loader=torch.utils.data.DataLoader(eval_dataset,batch_size=126,shuffle=False,num_workers=4)

    ###Get the embeddings
    all_embeddings, all_labels, all_text, all_paths=get_image_text_embeddings(eval_loader,clip_model,None,device,processor,"mean",0.5)

    ###Pairwise distances using faiss - gpu
    ###Get the pairwise distances

    res=faiss.StandardGpuResources()
    D=faiss.pairwise_distance_gpu(res, all_embeddings, all_embeddings)
    D = np.sort(D, axis=1)

    print(D)





    

