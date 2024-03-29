####Continue pretrain clip
###This script is to continue pretraining clip on the japanese dataset. Our dataset is a df of image-text pairs

import pandas as pd
import numpy as np
import wandb
import faiss 
from tqdm import tqdm
import argparse
import torch
from pytorch_metric_learning import losses
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, CosineAnnealingLR, StepLR
from torch import nn
import wandb
from transformers import CLIPProcessor, CLIPModel
import models.encoders as encoders
from sklearn.metrics import adjusted_mutual_info_score, rand_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from hyperopt import hp,fmin, tpe


from utils.datasets_utils import *
import datasets.data_loaders as data_loaders





def convert_to_text(unicode_string):
    return unicode_string.encode('ascii','ignore').decode('ascii')

def prep_labelled_news_data(singletons_only=True):
    ###Load the text file with the labels
    if singletons_only:
        train_data = pd.read_csv(f'/path/to/data/clippings_general/texts/labelled_news_train_reformatted_no_singletons.csv')
        val_data = pd.read_csv(f'/path/to/data/clippings_general/texts/labelled_news_val_reformatted_no_singletons.csv')

    ##With singletons
    # train_data = pd.read_csv(f'/path/to/data/clippings_general/texts/labelled_news_train_reformatted.csv')
    # val_data = pd.read_csv(f'/path/to/data/clippings_general/texts/labelled_news_val_reformatted.csv')
    
    return train_data,val_data

def prep_food101_data():
    ###Load the text file with the labels
    train_data = pd.read_csv(f'/path/to/data/clippings_general/texts/train_titles_reformatted.csv')
    val_data = pd.read_csv(f'/path/to/data/clippings_general/texts/val_titles_reformatted.csv')
    test_data = pd.read_csv(f'/path/to/data/clippings_general/texts/test_titles_reformatted.csv')

    return train_data, val_data, test_data

def prep_unlabelled_news_data():
    ###Load the text file with the labels
    train_data = pd.read_csv(f'/path/to/data/clippings_general/texts/unlabelled_news_train_reformatted.csv')
    val_data = pd.read_csv(f'/path/to/data/clippings_general/texts/unlabelled_news_val_reformatted.csv')

    return train_data,val_data

def eval_clip(val_loader,model,processor):
    print("Evaluating the model - clip loss")
    model.eval()
    loss_list=[]
    with torch.no_grad():
        for batch_idx, (text, image_data, labels, image_path) in enumerate(tqdm(val_loader)):
            labels = labels.to(device)
            image_data = image_data.to(device)
            labels= torch.arange((labels.shape[0])).to(device)

            text_features=(processor.tokenizer(text, return_tensors="pt", padding=True,max_length=77,truncation=True))

            for key in text_features.keys():
                text_features[key]=text_features[key].to(device)

            model_output=model.forward(input_ids=text_features["input_ids"],pixel_values=image_data,
                                                attention_mask=text_features["attention_mask"])
            logits_per_image, logits_per_text = model_output["logits_per_image"], model_output["logits_per_text"]

            loss = (img_loss(logits_per_image, labels) + text_loss(logits_per_text, labels))/2
            loss_list.append(loss.item())

    mean_loss= np.mean(loss_list)
    wandb.log({"val_loss":mean_loss})
    return mean_loss            




def pretrain_clip(train_loader,model,device,img_loss,text_loss,epoch,optimizer,processor,scheduler=None,epochviz=None):
    print("Pretraining CLIP")
    """REf: https://github.com/openai/CLIP/blob/main/clip/model.py | https://github.com/openai/CLIP/issues/83 """
    # model.to(device)
    model.train()

    loss_list=[]
    for batch_idx, (text, image_data, labels, image_path) in tqdm(enumerate(train_loader)):
        labels = labels.to(device)

        ####Unsquueze the image data
        image_data = image_data.to(device)

        ### text is a tuple of strings, we need to convert it to a tensor
        text_features=processor.tokenizer(text=text, return_tensors="pt", padding=True,max_length=77,truncation=True)

        for key in text_features.keys():
            text_features[key]=text_features[key].to(device)

        
        optimizer.zero_grad()


        model_output=model.forward(input_ids=text_features["input_ids"],pixel_values=image_data,attention_mask=text_features["attention_mask"])

        logits_per_image, logits_per_text = model_output["logits_per_image"], model_output["logits_per_text"]
        del model_output

        ###The clip objective asks us to maximize the similarity between the logits of the image and text. Labels aren't needed here.
        ###Giving them a diff label for each image and text pair
        labels=torch.arange((labels.shape[0]))
        labels=labels.to(device)

        loss = (img_loss(logits_per_image, labels) + text_loss(logits_per_text, labels))/2

        loss.backward()

        optimizer.step()

        ##For ReduceLROnPlateau scheduler, we need to pass the loss value
        if scheduler!=None:
            scheduler.step()
            # scheduler.step(loss.item())
            if batch_idx % 50 == 0:
                print("Current LR: {}".format(scheduler.get_lr()[0]))
            wandb.log({"train/lr": scheduler.get_lr()[0]})
        wandb.log({"train/loss": loss.item()})

        if batch_idx % 50 == 0:
            print("Epoch {} Iteration {}: Loss = {}".format(
                str(epoch).zfill(3), str(batch_idx).zfill(4), loss))
            if not epochviz is None:
                for i in range(30):
                    image = T.ToPILImage()(INV_NORMALIZE(image_data[i].cpu()))
                    image.save(os.path.join(epochviz, f"train_sample_{epoch}_{i}.png"))

        loss_list.append(loss.item())
    
    ####Mean epoch loss
    mean_epoch_loss=np.mean(loss_list)
    return mean_epoch_loss



def train_bienc_clip(train_loader,clip_model,device,loss_func,epoch,clip_optimizer,processor,clip_scheduler=None,epochviz=None,mlp_model=None,mlp_optimizer=None,mlp_scheduler=None,freeze_clip=False):
    """A version where we contrastively train pooled clip embeddings"""


    clip_model.train()
    if not mlp_model is None:
        mlp_model.train()


    for batch_idx, (text, image_data, labels, image_path) in tqdm(enumerate(train_loader)):

        labels = labels.to(device)

       ####Unsquueze the image data
        image_data = image_data.to(device)

        ### text is a tuple of strings, we need to convert it to a tensor
        text_features=processor.tokenizer(text=text, return_tensors="pt", padding=True,max_length=77,truncation=True)

        for key in text_features.keys():
            text_features[key]=text_features[key].to(device)

        
        clip_optimizer.zero_grad()
        if not mlp_model is None:
            mlp_optimizer.zero_grad()


        if freeze_clip:
            with torch.no_grad():
                model_output=clip_model.forward(input_ids=text_features["input_ids"],pixel_values=image_data,attention_mask=text_features["attention_mask"])
        
        else:
                model_output=clip_model.forward(input_ids=text_features["input_ids"],pixel_values=image_data,attention_mask=text_features["attention_mask"])
        image_embeds, text_embeds = model_output["image_embeds"], model_output["text_embeds"]
        del model_output

        if args.pooling_type=="mean":
            final_embeds= args.im_wt*image_embeds + (1-args.im_wt)*text_embeds
        elif args.pooling_type=="mlp":
            ###Use an MLP to combine the image and text embeddings
            ###concat the image and text embeddings
            final_embeds=torch.cat([image_embeds,text_embeds],dim=1)
            ###Pass through an MLP
            final_embeds=mlp_model.forward(final_embeds)
        else:
            raise ValueError("Pooling type not supported")
        
        
        ##L2 normalize the embeddings
        final_embeds=torch.nn.functional.normalize(final_embeds,p=2,dim=1)

        loss=loss_func(final_embeds,labels)

        loss.backward()

        clip_optimizer.step()
        if not mlp_optimizer is None:
            mlp_optimizer.step()

        ##For ReduceLROnPlateau scheduler, we need to pass the loss value
        if clip_scheduler!=None:
            clip_scheduler.step()
            # scheduler.step(loss.item())
            if batch_idx % 50 == 0:
                print("Current LR: {}".format(clip_scheduler.get_lr()[0]))
            wandb.log({"train/clip_lr": clip_scheduler.get_lr()[0]})
        
        if mlp_scheduler!=None:
            mlp_scheduler.step()
            # scheduler.step(loss.item())
            if batch_idx % 50 == 0:
                print("Current LR: {}".format(mlp_scheduler.get_lr()[0]))
            wandb.log({"train/mlp_lr": mlp_scheduler.get_lr()[0]})

        wandb.log({"train/loss": loss.item()})

        if batch_idx % 50 == 0:
            print("Epoch {} Iteration {}: Loss = {}".format(
                str(epoch).zfill(3), str(batch_idx).zfill(4), loss))
            if not epochviz is None:
                for i in range(10):
                    image = T.ToPILImage()(INV_NORMALIZE(image_data[i].cpu()))
                    image.save(os.path.join(epochviz, f"train_sample_{epoch}_{i}.png"))


def train_bienc_classifier(num_classes=0):
    pass
        

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


def tester_bienc_clip(test_loader,ref_loader,clip_model,mlp_model,split='val',log=True):
    print("Testing using pooled embeddings")

    
    test_embeddings, test_labels, test_text, test_paths = get_image_text_embeddings(test_loader,clip_model,mlp_model,device,processor,args.pooling_type,args.im_wt)
    print("total test embeddings: ",test_embeddings.shape)
    ref_embeddings, ref_labels, ref_text, ref_paths = get_image_text_embeddings(ref_loader,clip_model,mlp_model, device,processor,args.pooling_type,args.im_wt)
    print("total ref embeddings: ",ref_embeddings.shape)
    ###Make an index
    index = faiss.IndexFlatIP(test_embeddings.shape[1])
    index.add(ref_embeddings.cpu().numpy())

    ###Get the nearest neighbours
    D, I = index.search(test_embeddings.cpu().numpy(), 1)


    acc=0
    for i in range(len(test_labels)):
        if test_labels[i]==ref_labels[I[i][0]]:
            acc+=1
    acc=acc/len(test_labels)
    print("CUSTOM ACCURACY: ",acc)


    if log:
        wandb.log({f"{split}/precision_1": acc})

    ###Print a sample of predictions (text)
    for i in range(10):
        print(f"Text: {test_text[i]}")
        print(f"Nearest neighbour: {ref_text[I[i][0]]}")
        print(f"Nearest neighbour label: {ref_labels[I[i][0]]}")
        print(f"Test label: {test_labels[i]}")
        print("")
        print(acc)

    return acc


def val_bienc_clip_loss(val_loader,clip_model,mlp_model,loss_fn,split='val',log=True,processor=None):
    print("Testing using pooled embeddings")

    
    test_embeddings, test_labels, test_text, test_paths = get_image_text_embeddings(val_loader,clip_model,mlp_model, device,processor,args.pooling_type,args.im_wt)
    print("total test embeddings: ",test_embeddings.shape)

    val_loss=loss_fn(test_embeddings,test_labels)


    if log:
        wandb.log({f"{split}/val loss": val_loss})

    return val_loss



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



    return cluster_assignment




def val_bienc_clustering(val_loader,clip_model,mlp_model,split='val',log=True,processor=None):
    print("Testing using pooled embeddings (clustering-ari)")
    
    test_embeddings, test_labels, test_text, test_paths = get_image_text_embeddings(val_loader,clip_model,mlp_model, device,processor,args.pooling_type,args.im_wt)
    print("total test embeddings: ",test_embeddings.shape)

    ###Split the embeddings and labels into train and val
    idx_val=np.random.choice(len(test_embeddings),int(len(test_embeddings)*0.2),replace=False)
    idx_train=np.setdiff1d(np.arange(len(test_embeddings)),idx_val)

    all_embeddings_val=torch.cat((test_embeddings[idx_train],test_embeddings[idx_val]))
    all_labels_val=torch.cat((test_labels[idx_train],test_labels[idx_val]))

    all_embeddings_test=test_embeddings[idx_val]
    all_labels_test=test_labels[idx_val]

    ###Cluster the val embeddings
    ###Use hyperopt to max the adjusted rand index
    def hyp_ari(params,all_embeddings=all_embeddings_val, all_labels=all_labels_val):
        print("Params",params)

        print("Get knn")

        all_embeddings=all_embeddings.cpu().numpy()
        all_labels=all_labels.cpu().numpy()
        print(all_labels)

        val_clusters=cluster("SLINK",cluster_params={"min cluster size":1,"threshold":params["threshold"],"metric":"cosine"},corpus_embeddings=all_embeddings,corpus_ids=None)
        print("ARI",adjusted_rand_score(all_labels,val_clusters))
        return -adjusted_rand_score(all_labels,val_clusters)
    
    space = {
        "threshold":hp.uniform("threshold",0.01,0.4),
        
    }

    best = fmin(hyp_ari, space, algo=tpe.suggest, max_evals=500)
    print(best) 

    all_embeddings_test=all_embeddings_test.cpu().numpy()
    all_labels_test=all_labels_test.cpu().numpy()


    cluster_preds=cluster("SLINK",cluster_params={"min cluster size":1,"threshold":best["threshold"],"metric":"cosine"},corpus_embeddings=all_embeddings_test,corpus_ids=None)

    ari_split=adjusted_rand_score(all_labels_test,cluster_preds)
    if log:
        wandb.log({f"{split}/ari": ari_split})

    return ari_split



if __name__ == "__main__":

        ##parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--clip_lr", type=float, default=5e-7)
    parser.add_argument("--mlp_lr", type=float, default=5e-5)
    parser.add_argument("--clip_weight_decay",type=float,default=0.001)
    parser.add_argument("--mlp_weight_decay",type=float,default=0.001)
    parser.add_argument("--batch_size",type=int,default=153)
    parser.add_argument("--m",type=int,default=1)
    parser.add_argument("--k",type=int,default=3)
    parser.add_argument("--train_data_type",type=str,default="labelled")
    parser.add_argument("--wandb_name",type=str,default="clip_pretrain_labelled_m1")
    parser.add_argument("--training_type",type=str,default="pretrain")
    parser.add_argument("--supcon_temp",type=float,default=0.1)
    parser.add_argument("--im_wt",type=float,default=0.3)
    parser.add_argument("--pooling_type",type=str,default="mean")
    parser.add_argument("--freeze_clip_epochs",type=int,default=20)
    parser.add_argument("--mlp_layers",type=int,default=3)
    parser.add_argument("--augmented_crops",action="store_true")
    parser.add_argument("--train_hardneg",action="store_true")
    parser.add_argument("--checkpoint_path",type=str,default=None)
    parser.add_argument("--loss_type",type=str,default="supcon")
    parser.add_argument("--contrastive_loss_pos_margin",type=float,default=0)
    parser.add_argument("--contrastive_loss_neg_margin",type=float,default=0.5)

    args = parser.parse_args()
    
    # def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    ###Load checkpoint
    if args.checkpoint_path is not None:
        clip_model.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device(device)))
    # model.load_state_dict(torch.load("/path/to/savedir/clippings_general/models/bienc_clip_pretrain_labelled_m3.pt", map_location=torch.device(device)))
    clip_model.to(device)

    if args.pooling_type=="mlp":
        mlp_model=encoders.MLP(2 * 512, 1024, 512, args.mlp_layers, 0.1)
        mlp_model.to(device)
    else:
        mlp_model=None




    print("Train data type: ",args.train_data_type)
    if args.train_data_type == "food101_labelled":
        train_data,val_data,test_data=prep_food101_data()
    elif args.train_data_type == "food101_unlabelled":
        train_data,val_data,test_data=prep_food101_data()
    elif args.train_data_type == "newspapers_unlabelled":
        train_data,val_data=prep_unlabelled_news_data()
        test_data=val_data
    elif args.train_data_type == "newspapers_labelled":
        train_data,val_data=prep_labelled_news_data()       
        test_data=val_data
    else:
        print("Not implemented yet")
        pass
    ###Prototype sample 1000
    # train_data=train_data.sample(n=1000,random_state=42)
    if args.training_type == "pretrain":
        val_data=val_data.sample(n=5000,random_state=42)
    else:
        pass   
    ###Remove any unnamed columns
    train_data=train_data.loc[:, ~train_data.columns.str.contains('^Unnamed')]
    val_data=val_data.loc[:, ~val_data.columns.str.contains('^Unnamed')]

    

    ###We wil drop duplicates in the train data if pretraining
    if args.training_type == "pretrain":
        ###Shuffle first
        print("Lenth of train data before dropping duplicates: ",len(train_data))
        train_data=train_data.sample(frac=1,random_state=42)
        train_data=train_data.drop_duplicates(subset=['text'],keep='first')
        print("Lenth of train data after dropping duplicates: ",len(train_data))
    
    
   

    ###Create the data datsets
    if args.augmented_crops:
        train_image_transform=create_clip_random_doc_transform()
    else:
        train_image_transform=CLIP_BASE_TRANSFORM_CENTER
    if args.train_hardneg:
        print("Setting up dataset with hardnegatives")
        dedup_train_data=train_data.drop_duplicates(subset=['label'],keep='first')
        print("Total number of unique labels in train data: ",len(dedup_train_data))
        k_hardneg_df = data_loaders.make_hard_neg_df(dedup_train_data,k=args.k,clip_model=clip_model,mlp_model=mlp_model,device=device,processor=processor,pooling_type=args.pooling_type,im_wt=args.im_wt)
        ##Save the hardneg df
        k_hardneg_df.to_csv("k_hardneg_df.csv") ###Use TextImageDatasetWithHardNegsSingle for incorporating singletons better TextImageDatasetWithHardNegs ow
        
        if args.loss_type == "contrastive":
            train_dataset=data_loaders.TextImageDatasetWithHardNegsSingle(train_data,k_hardneg_df,img_transform=  train_image_transform ,text_transform=None,batch_size=args.batch_size,k=args.k,m=args.m)
        else : 
            train_dataset=data_loaders.TextImageDatasetWithHardNegs(train_data,k_hardneg_df,img_transform=  train_image_transform ,text_transform=None,batch_size=args.batch_size,k=args.k,m=args.m)
        print("Done setting up dataset with hardnegatives")
    else: 
        train_dataset=data_loaders.TextImageDataset(train_data, img_transform=train_image_transform)
    
    val_dataset=data_loaders.TextImageDataset(val_data,img_transform=CLIP_BASE_TRANSFORM_CENTER)
    if args.training_type != "pretrain":
        test_dataset=data_loaders.TextImageDataset(test_data,img_transform=CLIP_BASE_TRANSFORM_CENTER)

    print(len(train_dataset))


    ###Create the data loaders
    if args.train_data_type == "food101_labelled" or args.train_data_type == "newspapers_labelled" :
        
        if args.train_hardneg:
            train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=False,num_workers=4)
        else:
            train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,sampler=data_loaders.NoReplacementMPerClassSampler(train_dataset, m=args.m,batch_size=args.batch_size,num_passes=1))

        val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    elif args.train_data_type == "food101_unlabelled" or args.train_data_type == "newspapers_unlabelled":
        if args.train_hardneg:
            train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=False,num_workers=4)
        else:
            train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
   
    else:
        raise ValueError("labelled_data must be either food101_labelled, food101_unlabelled or newspapers")
    if args.training_type != "pretrain":
        test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    ###ADditonally, if training biencoder with synthetic data, create a reference dataset

    ###Set up device
    # setup

    img_loss=nn.CrossEntropyLoss()
    text_loss=nn.CrossEntropyLoss()


    ###Optimizer for both clip and mlp
    clip_optimizer = torch.optim.AdamW(clip_model.parameters(), lr=args.clip_lr,weight_decay=args.clip_weight_decay, betas=(0.9,0.98),
                    eps=1e-06)
    clip_scheduler = CosineAnnealingWarmRestarts(clip_optimizer, 10, 2)

    if args.pooling_type=="mlp":
        mlp_optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=args.mlp_lr,weight_decay=args.mlp_weight_decay, betas=(0.9,0.98),
                        eps=1e-06)
        mlp_scheduler = CosineAnnealingWarmRestarts(mlp_optimizer, 10, 2)
    
    else :
        mlp_optimizer=None
        mlp_scheduler=None



    ###Set up the trainer
    wandb.init(project="multimodal_record_linkage", name=args.wandb_name)
    num_epochs=1000
    start_epoch=0
    best_acc=0

    if args.loss_type=="supcon":
        loss_func=losses.SupConLoss(temperature=args.supcon_temp)
    elif args.loss_type=="contrastive":
        loss_func=losses.ContrastiveLoss(pos_margin=args.contrastive_loss_pos_margin, neg_margin=args.contrastive_loss_neg_margin)
    else:
        ValueError("Contrastive loss must be either supcon or contrastive")



    if args.training_type=="pretrain":
        zero_shot_loss=eval_clip(val_loader,clip_model,processor)
    else:
        pass

    if args.training_type=="pretrain":


        for epoch in (range(start_epoch, num_epochs+start_epoch)):
            train_loss=pretrain_clip(train_loader,clip_model,device,img_loss,text_loss,epoch,clip_optimizer,processor,clip_scheduler,epochviz=None)
            if epoch>0:
                val_loss=eval_clip(val_loader,clip_model,processor)
            # print("val Accuracy: {}".format(acc))
            # acc=tester_bienc_clip(val_loader,val_loader,model,split="val_small",log=True)
                print("Val loss: {}".format(val_loss))
                print("Train loss: {}".format(train_loss))

                if val_loss<zero_shot_loss:
                    zero_shot_loss=val_loss
                    torch.save(clip_model.state_dict(), os.path.join("/path/to/savedir/clippings_general/models/",args.wandb_name+".pt"))
                    
                    print("Model saved at epoch {}".format(epoch))
                    print("Path of the saved model: {}".format(os.path.join("/path/to/savedir/clippings_general/models/",args.wandb_name+".pt")))
                    print("Path of the saved model: {}".format(os.path.join("/path/to/savedir/clippings_general/models/",("epoch_"+str(epoch)+"_"+args.wandb_name+".pt"))))
                    print("Val loss: {}".format(val_loss))
                    if val_loss<0.1:
                        torch.save(clip_model.state_dict(), os.path.join("/path/to/savedir/clippings_general/models/",("epoch_"+str(epoch)+args.wandb_name+".pt")))
                ##Also save the model at the end of each epoch
                # torch.save(clip_model.state_dict(), os.path.join("/path/to/savedir/clippings_general/models/",("epoch_"+str(epoch)+args.wandb_name+".pt")))

    elif args.training_type=="train_bienc" and args.train_data_type=="newspapers_labelled":
        best_val_ari=val_bienc_clustering(val_loader,clip_model,mlp_model,split='val',log=True,processor=processor)
        print("Best Val ARI: {}".format(best_val_ari))
        # best_acc=tester_bienc_clip(val_loader,huge_ref_loader,clip_model,mlp_model,split="val_huge",log=True)
        for epoch in (range(start_epoch, num_epochs+start_epoch)):
            if epoch<= args.freeze_clip_epochs:
                if args.pooling_type=="mlp":
                    freeze_clip=True
                else:
                    freeze_clip=False
                epoch_loss=train_bienc_clip(train_loader,clip_model,device,loss_func,epoch,clip_optimizer,clip_scheduler=clip_scheduler,epochviz="/path/to/savedir/clippings_general/epoch_viz/",processor=processor,mlp_model=mlp_model,mlp_optimizer=mlp_optimizer,mlp_scheduler=mlp_scheduler,freeze_clip=freeze_clip)
            else:
                freeze_clip=False
                epoch_loss=train_bienc_clip(train_loader,clip_model,device,loss_func,epoch,clip_optimizer,clip_scheduler=clip_scheduler,epochviz="/path/to/savedir/clippings_general/epoch_viz/",processor=processor,mlp_model=mlp_model,mlp_optimizer=mlp_optimizer,mlp_scheduler=mlp_scheduler,freeze_clip=freeze_clip)
            
            val_ari=val_bienc_clustering(val_loader,clip_model,mlp_model,split='val',log=True,processor=processor)
            if val_ari>best_val_ari:
                best_val_ari=val_ari
                torch.save(clip_model.state_dict(), os.path.join("/path/to/savedir/clippings_general/models/",("clip_imwt_"+str(args.im_wt)[2]+args.wandb_name+".pt")))
                print("Model saved at epoch {}".format(epoch))
                print("Path of the saved model: {}".format(os.path.join("/path/to/savedir/clippings_general/models/",("clip_imwt_"+str(args.im_wt)[2]+args.wandb_name+".pt"))))
                if args.pooling_type=="mlp":
                    torch.save(mlp_model.state_dict(), os.path.join("/path/to/savedir/clippings_general/models/",("mlp_imwt_"+str(args.im_wt)[2]+args.wandb_name+".pt")))
                print("Model saved at epoch {}".format(epoch))
            ###SAve at every epoch
            # torch.save(clip_model.state_dict(), os.path.join("/path/to/savedir/clippings_general/models/",("clip_imwt_"+str(args.im_wt)[2]+"epoch_"+str(epoch)+args.wandb_name+".pt")))
    
       
    else :
        print("Training type not recognised")
        raise ValueError





