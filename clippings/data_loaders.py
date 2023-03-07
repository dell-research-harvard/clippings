####Train a multi-modal siamese network for image and text embeddings
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torchvision.transforms as T
from pytorch_metric_learning.utils import common_functions as c_f

import numpy as np
import pandas as pd
from timm.models import load_state_dict



import timm

import faiss
import wandb

import os
import sys
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from utils.datasets_utils import *

from tqdm import tqdm
from utils.datasets_utils import *







###Dataset for text and image data. Each sample is a tuple of (text, image)
###The data is stored in paths in a csv file. The csv is loaded seperately into a pandas df
## . Each row contains paired images and their corresponding ocr texts, and a label
###Return a tuple of (text, image, target) where target just corresponds to label!

def clean_checkpoint(checkpoint, use_ema=True, clean_aux_bn=False,clean_net=True):
    # Load an existing checkpoint to CPU, strip everything but the state_dict and re-save
    if checkpoint and os.path.isfile(checkpoint):
        print("=> Loading checkpoint '{}'".format(checkpoint))
        state_dict = load_state_dict(checkpoint, use_ema=use_ema)
        new_state_dict = {}
        for k, v in state_dict.items():
            if clean_aux_bn and 'aux_bn' in k:
                # If all aux_bn keys are removed, the SplitBN layers will end up as normal and
                # load with the unmodified model using BatchNorm2d.
                continue
            # name = k[7:] if k.startswith('module.') else k
            # new_state_dict[name] = v
            
            name = k[7:] if (k.startswith('module.') and clean_net==True) else k
            new_state_dict[name] = v
        print("=> Loaded state_dict from '{}'".format(checkpoint))
        return new_state_dict
    else:
        print("Error: Checkpoint ({}) doesn't exist".format(checkpoint))
        return ''

class TextImageDataset(Dataset):

    def __init__(self,pandas_df,img_transform=None,text_transform=None):

        self.df = pandas_df
        self.img_transform = img_transform
        self.text_transform = text_transform

        self.df = self.df.dropna()

        self.targets = self.df['label'].values


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
            
            if torch.is_tensor(idx):
                idx = idx.tolist()
    
            img_path = self.df.iloc[idx,0]
            text = self.df.iloc[idx,1]
            label=self.df.iloc[idx,2]
    
            img = Image.open(img_path)
    
            if self.img_transform:
                img = self.img_transform(img)
    
            if self.text_transform:
                text = self.text_transform(text)
    
    
            return text, img, label, img_path

###Dataloader for text and image data. Each sample is a tuple of (text, image,target). Takes in the dataset and the batch size.

class TextImageDataLoader(DataLoader):
    
        def __init__(self,dataset,batch_size=64,shuffle=True,num_workers=32):
    
            super().__init__(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)

    

###Sampler
class NoReplacementMPerClassSampler(Sampler):

    def __init__(self, dataset, m, batch_size, num_passes):
        labels = dataset.targets
        # print(labels)
        assert not batch_size is None, "Batch size is None!"
        if isinstance(labels, torch.Tensor): labels = labels.numpy()
        self.m_per_class = int(m)
        self.batch_size = int(batch_size)
        self.labels_to_indices = c_f.get_labels_to_indices(labels)
        self.labels = list(self.labels_to_indices.keys())
        self.length_of_single_pass = self.m_per_class * len(self.labels)
        self.dataset_len = int(self.length_of_single_pass * num_passes) # int(math.ceil(len(dataset) / batch_size)) * batch_size
        assert self.dataset_len >= self.batch_size
        assert self.length_of_single_pass >= self.batch_size, f"m * (number of unique labels ({len(self.labels)}) must be >= batch_size"
        assert self.batch_size % self.m_per_class == 0, "m_per_class must divide batch_size without any remainder"
        self.dataset_len -= self.dataset_len % self.batch_size

    def __len__(self):
        return self.dataset_len

    def __iter__(self):

        idx_list = [0] * self.dataset_len
        i = 0; j = 0
        num_batches = self.calculate_num_batches()
        num_classes_per_batch = self.batch_size // self.m_per_class
        c_f.NUMPY_RANDOM.shuffle(self.labels)

        indices_remaining_dict = {}
        for label in self.labels:
            indices_remaining_dict[label] = set(self.labels_to_indices[label])

        for _ in range(num_batches):
            curr_label_set = self.labels[j : j + num_classes_per_batch]
            j += num_classes_per_batch
            assert len(curr_label_set) == num_classes_per_batch, f"{j}, {len(self.labels)}"
            if j + num_classes_per_batch >= len(self.labels):
                print(f"All unique labels/classes batched, {len(self.labels)}; restarting...")
                c_f.NUMPY_RANDOM.shuffle(self.labels)
                j = 0
            for label in curr_label_set:
                t = list(indices_remaining_dict[label])
                if len(t) == 0:
                    randchoice = c_f.safe_random_choice(self.labels_to_indices[label], size=self.m_per_class).tolist()
                elif len(t) < self.m_per_class:
                    randchoice = t + c_f.safe_random_choice(self.labels_to_indices[label], size=self.m_per_class-len(t)).tolist()
                else:
                    randchoice = c_f.safe_random_choice(t, size=self.m_per_class).tolist()
                indices_remaining_dict[label] -= set(randchoice)
                idx_list[i : i + self.m_per_class] = randchoice
                i += self.m_per_class
        
        notseen_count = 0
        for k in indices_remaining_dict.keys():
            notseen_count += len(indices_remaining_dict[k])
        print(f"Samples not seen: {notseen_count}")

        return iter(idx_list)

    def calculate_num_batches(self):
        assert self.batch_size < self.dataset_len, "Batch size is larger than dataset!"
        return self.dataset_len // self.batch_size

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
        text_features = processor.tokenizer(text)

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


###Dataset with hardnegs. Each sample is a tuple of (text, image, label, anchor_id)
###The data is stored in paths in a csv file. The csv is loaded seperately into a pandas df
## . Each row contains paired images and their corresponding ocr texts, and a label

def make_hard_neg_df(original_df,k,clip_model,mlp_model,device,processor,pooling_type="mean",im_wt=0.5,save_path=None):
    """Load as dataset,loader, get embeddings and construct a df with k nn for each sample and give all of them a unique anchor_id"""    

    ###pre process for labelled data
    
    original_dataset = TextImageDataset(original_df,img_transform=CLIP_BASE_TRANSFORM)
    original_loader = TextImageDataLoader(original_dataset,batch_size=128,shuffle=False,num_workers=32)

    ##Get embeddings
    all_embeddings, all_labels, all_text, all_paths = get_image_text_embeddings(original_loader,clip_model,mlp_model,device,processor,pooling_type,im_wt)

    ###Get k nearest neighbors for each sample
    ##Make FAISS index
    all_embeddings=all_embeddings.cpu().numpy()
    index = faiss.IndexFlatL2(all_embeddings.shape[1])
    index.add(all_embeddings)

    ##Get k nearest neighbors
    D,I = index.search(all_embeddings,k=k)

    ###Construct a df with k nn for each sample and give all of them a unique anchor_id
    ##Give anchor ids to each ID

    ##Get k nearest neighbor texts
    knn_texts = []
    knn_img_paths = []
    knn_labels = []
    knn_anchor_ids = []
    for i in range(len(all_text)):
        knn_texts.append(np.array(all_text)[I[i].astype(int)])
        knn_img_paths.append(np.array(all_paths)[I[i].astype(int)])
        knn_labels.append(np.array(all_labels.cpu())[I[i].astype(int)])
        knn_anchor_ids.append(np.array([i]*k))


    
    

    ##Construct df
    knn_df = pd.DataFrame()
    knn_df["img_path"] = np.concatenate(knn_img_paths)
    knn_df["text"] = np.concatenate(knn_texts)
    knn_df["label"] = np.concatenate(knn_labels)
    knn_df["anchor_id"] = np.concatenate(knn_anchor_ids)

    return knn_df





class TextImageDatasetWithHardNegs(Dataset):
    
        def __init__(self,original_df,k_hardneg_df,img_transform=None,text_transform=None,batch_size=126,k=3,m=3):
            
            ##Original dataset length


            ###Batch size needs to be divisible by k*m
            assert batch_size % (k*m) == 0, "Batch size must be divisible by k*m"
            self.batch_size = batch_size
            self.k = k
            self.m = m
            self.k_hardneg_df = k_hardneg_df
            self.img_transform = img_transform
            self.text_transform = text_transform
            self.original_df = original_df
    
            self.k_hardneg_df = self.k_hardneg_df.dropna()
            ##Get unique anchor ids
            self.anchor_ids = self.k_hardneg_df.anchor_id.unique()

            ###Partition anchor ids into batch_size/k*m groups
            self.anchor_id_groups = np.array_split(self.anchor_ids,self.batch_size/(self.k*self.m))
            self.anchor_id_groups = [list(x) for x in self.anchor_id_groups]

            ###Now, for each anchor id, get a unique list of labels
            self.anchor_id_to_labels = {}
            for anchor_id in self.anchor_ids:
                self.anchor_id_to_labels[anchor_id] = self.k_hardneg_df[self.k_hardneg_df.anchor_id == anchor_id].label.unique()
            
            ####Now for each label in each anchor id, sample m rows corresponding to that label
            self.anchor_id_to_label_to_rows = {}
            for anchor_id in self.anchor_ids:
                self.anchor_id_to_label_to_rows[anchor_id] = {}
                for label in self.anchor_id_to_labels[anchor_id]:
                    self.anchor_id_to_label_to_rows[anchor_id][label] = self.original_df[ (self.original_df.label == label)].sample(m,replace=True)

            ###Now, reconstruct the dataframe
            self.df = pd.DataFrame()
            for anchor_id in self.anchor_ids:
                for label in self.anchor_id_to_labels[anchor_id]:
                    self.df = self.df.append(self.anchor_id_to_label_to_rows[anchor_id][label])
                    ###Add anchor_id to each row
                    self.df.loc[self.df.label == label,"anchor_id"] = anchor_id
            
            ###Name the columns
          

            self.df = self.df.reset_index(drop=True)
            self.df = self.df.dropna()

            print("Exapanded Dataset length: ",len(self.df))
            print("Original Dataset length: ",len(original_df))
            print(self.df.head(10))

    
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self,idx):
                
                if torch.is_tensor(idx):
                    idx = idx.tolist()
        
                img_path = self.df.iloc[idx,0]
                text = self.df.iloc[idx,1]
                label=self.df.iloc[idx,2]
                anchor_id=self.df.iloc[idx,3]
        
                img = Image.open(img_path)
        
                if self.img_transform:
                    img = self.img_transform(img)
        
                if self.text_transform:
                    text = self.text_transform(text)
        
        
                return text, img, label, anchor_id
        


###Run as script
# if __name__ == "__main__":
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    

    # clip_model, preprocess = ja_clip.load("rinna/japanese-clip-vit-b-16", cache_dir="/tmp/japanese_clip", device=device)
    # cp_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/homoglyphs/multimodal_record_linkage/models/imwt_5bienc_clip_pretrain_labelled_m3_v3.pt"
    # mlp_model=None
    # clip_model.load_state_dict((clean_checkpoint(cp_path)))
    # clip_model.to(device)

    # tokenizer=ja_clip.load_tokenizer()

    # original_df_path = "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/multimodal_data/PR_TK_matched_ocr_only_train.csv"
    
    # original_df=pd.read_csv(original_df_path)
    # ###drop duplicates if image path and text are the same
    # original_df=original_df.drop_duplicates(subset=['image_path','text'])

    # ###Drop duplicates by "pr" and "text". Image is "pr" if image_path contains dot_dect_1130/element_crop/

    # ##Create a pr variable 1 if image_path contains dot_dect_1130/element_crop/
    # original_df['pr']=original_df['image_path'].apply(lambda x: 1 if "dot_dect_1130/element_crop/" in x else 0)

    # ##Drop duplicate rows when pr=1 and text is the same. 
    # split_data=original_df[original_df['pr']==1].drop_duplicates(subset=['text'])

    # ##Merge the split data with the data where pr=0
    # original_df=pd.concat([split_data,original_df[original_df['pr']==0]])

    # print("post processing size of data: {}".format(len(original_df)))
    # ##Drop pr
    # original_df=original_df.drop(columns=['pr'])

    # ###GEt only 1 view of each image
    # dedup_data=original_df.drop_duplicates(subset=['label'])

    # k_hardneg_df = make_hard_neg_df(dedup_data,k=3,clip_model=clip_model,mlp_model=mlp_model,device=device,tokenizer=tokenizer,pooling_type="mean",im_wt=5)

    # ##Save the df
    # k_hardneg_df.to_csv("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/multimodal_data/train_val_hardneg_data.csv",index=False,encoding="utf-8-sig")


    # ##Create the hadneg dataset
    # hardneg_dataset = TextImageDatasetWithHardNegs(original_df,k_hardneg_df,img_transform=CLIP_BASE_TRANSFORM,text_transform=None,batch_size=126,k=3,m=3)

    # ##Create the dataloader
    # hardneg_dataloader = DataLoader(hardneg_dataset,batch_size=126,shuffle=False,num_workers=4)

    # ##Test the dataloader
    # for i, (text, img, label, anchor_id) in enumerate(hardneg_dataloader):
    #     print("Batch: ",i)
    #     print("Text: ",text)
    #     print("Image: ",img)
    #     print("Label: ",label)
    #     print("Anchor ID: ",anchor_id)
    #     break
