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


from sklearn.metrics import adjusted_mutual_info_score, rand_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from hyperopt import hp,fmin, tpe




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


def cluster(cluster_type, cluster_params, corpus_embeddings, corpus_ids=None):

    if cluster_type not in ["agglomerative", "HDBScan", "SLINK"]:
        raise ValueError('cluster_type must be "agglomerative", "HDBScan", "community" or "SLINK"')
    if cluster_type == "agglomerative":
        if "threshold" not in cluster_params:
            raise ValueError('cluster_params must contain "threshold"')
        if "clustering linkage" not in cluster_params:
            raise ValueError('cluster_params must contain "clustering linkage"')
        if "metric" not in cluster_params:
            raise ValueError('cluster_params must contain "metric"')
    if cluster_type == "HDBScan":
        if "min cluster size" not in cluster_params:
            raise ValueError('cluster_params must contain "min cluster size"')
        if "min samples" not in cluster_params:
            raise ValueError('cluster_params must contain "min cluster size"')
    if cluster_type == "SLINK":
        if "min cluster size" not in cluster_params:
            raise ValueError('cluster_params must contain "min cluster size"')
        if "threshold" not in cluster_params:
            raise ValueError('cluster_params must contain "threshold"')


    if cluster_type == "agglomerative":
        clustering_model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=cluster_params["threshold"],
            linkage=cluster_params["clustering linkage"],
            affinity=cluster_params["metric"]
        )

    if cluster_type == "SLINK":
        clustering_model = DBSCAN(
            eps=cluster_params["threshold"],
            min_samples=cluster_params["min cluster size"],
            metric=cluster_params["metric"]
        )

    if cluster_type == "HDBScan":
        clustering_model = hdbscan.HDBSCAN(
            min_cluster_size=cluster_params["min cluster size"],
            min_samples=cluster_params["min samples"],
            gen_min_span_tree=True
        )

    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    # clustered_ids = {}
    # for sentence_id, cluster_id in enumerate(cluster_assignment):
    #     if int(cluster_id) not in clustered_ids:
    #         clustered_ids[int(cluster_id)] = []

    #     if corpus_ids:
    #         clustered_ids[int(cluster_id)].append(corpus_ids[sentence_id])
    #     else:
    #         clustered_ids[int(cluster_id)].append(sentence_id)

    # # HDBScan has a cluster where it puts all the unassigned nodes
    # if cluster_type == "HDBScan" or cluster_type == "SLINK" and -1 in clustered_ids:
    #     del clustered_ids[-1]

    return cluster_assignment


###Run as script
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--im_wt", type=float, default=0.5, help="Weight of image embeddings")
    parser.add_argument("--pooling_type", type=str, default="mean", help="Pooling type")

    args = parser.parse_args()
    # checkpoint_path="/mnt/data01/clippings_general/models/clip_pretrain_unlabelled_m1_newspapers_cc.pt"
    # checkpoint_path="/mnt/data01/clippings_general/models/clip_imwt_5bienc_clip_pretrain_labelled_m3_v3_newspapers_nosingle.pt"
    # checkpoint_path=None
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    clip_transform=CLIP_BASE_TRANSFORM_CENTER
    ###Load checkpoint
    if args.checkpoint_path is not None:
        clip_model.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device(device)))
    # model.load_state_dict(torch.load("/mnt/data01/clippings_general/models/bienc_clip_pretrain_labelled_m3.pt", map_location=torch.device(device)))
    clip_model.to(device)

    ###Load data
    test_data=pd.read_csv("/mnt/data01/clippings_general/texts/labelled_news_eval_reformatted.csv")
    test_data=test_data.sort_values(by="label")


    ###Eval data
    eval_data=pd.read_csv("/mnt/data01/clippings_general/texts/labelled_news_val_reformatted.csv")
    eval_data=eval_data.sort_values(by="label")



    ##Load the dataset
    ###Create the data datsets
    ###Tune params using eval set

    eval_dataset=data_loaders.TextImageDataset(eval_data, img_transform=clip_transform)
    eval_loader=torch.utils.data.DataLoader(eval_dataset,batch_size=126,shuffle=False,num_workers=16)

    ###Get the embeddings
    all_embeddings, all_labels, all_text, all_paths=get_image_text_embeddings(eval_loader,clip_model,None,device,processor,args.pooling_type,args.im_wt)

    print("Get knn")

    ###Build the index
    index = faiss.IndexFlatIP( 512)

    ###Add the embeddings
    index.add(all_embeddings.cpu().numpy())

    print("Done adding embeddings")

    ###Get the top 1000 nearest neighbours
    D, I = index.search(all_embeddings.cpu().numpy(),    all_embeddings.shape[0])


    all_embeddings=all_embeddings.cpu().numpy()
    all_labels=all_labels.cpu().numpy()
    print(all_labels)

    ###Get the clusters

    ###Use hyperopt to max the adjusted rand index
    def hyp_ari(params):
        print("Params",params)
        clusters=cluster("SLINK",cluster_params={"min cluster size":1,"threshold":params["threshold"],"metric":"cosine"},corpus_embeddings=all_embeddings,corpus_ids=None)
        print("Clusters",clusters)
        print("Max cluster",max(clusters))
        print("ARI",adjusted_rand_score(all_labels,clusters))
        return -adjusted_rand_score(all_labels,clusters)
    
    space = {
        "threshold":hp.uniform("threshold",0.01,0.4)
    }

    best = fmin(hyp_ari, space, algo=tpe.suggest, max_evals=1000)
    print(best)


    ###Now calculate test ARI using the best params
    ##First embed the test data
    test_dataset=data_loaders.TextImageDataset(test_data, img_transform=clip_transform)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=126,shuffle=False,num_workers=16)

    ###Get the embeddings
    all_embeddings, all_labels, all_text, all_paths=get_image_text_embeddings(test_loader,clip_model,None,device,processor,"mean",0.5)

    print("Get knn")

    ###Build the index
    index = faiss.IndexFlatIP( 512)

    ###Add the embeddings
    index.add(all_embeddings.cpu().numpy())

    print("Done adding embeddings")

    ###Get the top 1000 nearest neighbours
    D, I = index.search(all_embeddings.cpu().numpy(),    all_embeddings.shape[0])


    all_embeddings=all_embeddings.cpu().numpy()
    all_labels=all_labels.cpu().numpy()

    clusters=cluster("SLINK",cluster_params={"min cluster size":1,"threshold":best["threshold"],"metric":"cosine"},corpus_embeddings=all_embeddings,corpus_ids=None)

    print("threshold",best["threshold"])
    print("checkpoint",args.checkpoint_path)
    print("im_wt",args.im_wt)
    print("pooling_type",args.pooling_type)
    print("ARI",adjusted_rand_score(all_labels,clusters))



    
  
    