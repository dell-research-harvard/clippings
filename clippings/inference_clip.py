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

    checkpoint_path="/mnt/data01/clippings_general/models/clip_pretrain_unlabelled_m1_newspapers_cc.pt"
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

    ##Sort by label
    eval_data=eval_data.sort_values(by="label")

    ###Make ground truth pairs - take combinations of 2 image_paths for each label
    
    gt_pairs=[]
    ###for each label, get all image paths, then take combinations of 2 and add as a tuple
    unique_labels=eval_data["label"]
    print("Unique labels",unique_labels)

    for label in unique_labels:
        label_data=eval_data[eval_data["label"]==label]
        image_paths=label_data["image_path"]
        combinations_label=list(combinations(image_paths,2))
        gt_pairs=gt_pairs+combinations_label
    


        
        



    ##Load the dataset
    ###Create the data datsets
    eval_dataset=data_loaders.TextImageDataset(eval_data, img_transform=clip_transform)
    eval_loader=torch.utils.data.DataLoader(eval_dataset,batch_size=126,shuffle=False,num_workers=16)

    ###Get the embeddings
    all_embeddings, all_labels, all_text, all_paths=get_image_text_embeddings(eval_loader,clip_model,None,device,processor,"mean",0)

    ##Take a subset of embeddngs
    # all_embeddings=all_embeddings[0:10]


    ###Pairwise distances using faiss - gpu
    ###Get the pairwise distances
    print("Get knn")
    # res=faiss.StandardGpuResources()
    print(all_embeddings.shape)



    ###Build the index
    index = faiss.IndexFlatIP( 512)

    ###Add the embeddings
    index.add(all_embeddings.cpu().numpy())

    print("Done adding embeddings")

    ###Get the top 1000 nearest neighbours
    D, I = index.search(all_embeddings.cpu().numpy(),    all_embeddings.shape[0])

    ###Check the text of some of the nearest neighbours apart from the same image
    for i in range(0,3):
        print("Image",i)
        print("Text",all_text[i])
        print("Nearest neighbours")
        nn_of_i_2=I[i][1:3]
        for nn in nn_of_i_2:
                print("NN",nn)
                print("Text",all_text[nn])
                print("Path",all_paths[nn])
                print("Label",all_labels[nn])

  
    # for thresh in np.arange(0.1,0.95,0.01):
    #     print("Threshold",thresh)
    #     above_threshold = D > thresh
    #     ##If 0 embeddings are above threshold, then set stop
    #     if np.sum(above_threshold)==0:
    #         break

    #     upper_only = np.triu(np.ones((all_embeddings.shape[0], all_embeddings.shape[0])) - np.identity(all_embeddings.shape[0]))
    #     result = above_threshold * upper_only

    #     indices = [index for index, value in np.ndenumerate(result) if value]
    #     edges = [[all_paths[pair[0]], all_paths[pair[1]]] for pair in indices]

    #     # Build graph
    #     G = nx.Graph()
    #     G.add_edges_from(edges)

    #     # Community detection
    #     communities = nx_comm.louvain_communities(G, resolution=1)

    #     pred_pairs = []
    #     for i in range(len(communities)):
    #         pred_pairs.extend(combinations(list(communities[i]), 2))

    #     clustered_ids = {}
    #     for i in range(len(communities)):
    #         clustered_ids[i] = list(communities[i])



    #     pred_pairs = [list(p) for p in pred_pairs]

    #     print(f'{(all_embeddings.shape[0])} examples grouped into {len(communities)} clusters')
    #     # Evaluate
    #     set_preds = set(map(tuple, pred_pairs))
    #     set_gt = set(map(tuple, gt_pairs))

    #     # Metrics
    #     true_pos = [i for i in set_gt if i in set_preds or (i[1], i[0]) in set_preds]
    #     false_pos = [i for i in set_preds if i not in set_gt and (i[1], i[0]) not in set_gt]
    #     false_neg = [i for i in set_gt if i not in set_preds and (i[1], i[0]) not in set_preds]

    #     tps = len(true_pos)
    #     fps = len(false_pos)
    #     fns = len(false_neg)

    #     precision = tps / (tps + fps)
    #     recall = tps / (tps + fns)
    #     f_score = 2 * (precision * recall) / (precision + recall)

    #     # wrongs = []
    #     # for fn in false_neg:
    #     #     wrongs.append(fn[0])
    #     #     wrongs.append(fn[1])
    #     # wrongs = list(set(wrongs))

    #     # for w in wrongs:
    #     #     print(text_dict[w])
    #     #     print("**")

    #     print(precision, recall, f_score)
    #     print(tps, fps, fns)


        

