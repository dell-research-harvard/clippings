###Reoganise food 101 data

import os
import pandas as pd
import numpy as np
###Import test_train_split
from sklearn.model_selection import train_test_split
import json
import networkx as nx
from itertools import combinations



def clusters_from_edges(edges_list):
    """Identify clusters of passages given a dictionary of edges"""

    # clusters via NetworkX
    G = nx.Graph()
    G.add_edges_from(edges_list)
    sub_graphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    sub_graph_dict = {}
    for i in range(len(sub_graphs)):
        sub_graph_dict[i] = list(sub_graphs[i].nodes())

    return sub_graph_dict


def edges_from_clusters(cluster_dict):
    """
    Convert every pair in a cluster into an edge
    """
    cluster_edges = []
    for cluster_id in list(cluster_dict.keys()):
        art_ids_list = cluster_dict[cluster_id]
        edge_list = [list(comb) for comb in combinations(art_ids_list, 2)]
        cluster_edges.extend(edge_list)

    return cluster_edges
###Load all unlabelled data

###Reformat the newspaper images data
# ###Load the directory containig the csvs
# unlabelled_dir = '/mnt/data02/captions/unlabelled_pairs/'
# ##Load all the csvs
# unlabelled_data = pd.concat([pd.read_csv(unlabelled_dir + file) for file in os.listdir(unlabelled_dir)])
# ##Remove the index column
# unlabelled_data = unlabelled_data.drop(columns=['Unnamed: 0'])
# ###Add a column with label=999
# unlabelled_data['label'] = 999
# ##Rename the columns
# unlabelled_data.columns = ['image_path', 'text', 'label']

# ##Add base path to image path
# unlabelled_data['image_path'] = '/mnt/data02/captions/pulled_crops/' + unlabelled_data['image_path'] + '.png'

# ##Split in val and train
# train, val = train_test_split(unlabelled_data, test_size=0.2, random_state=42)

# ##Save the text data
# train.to_csv(f'texts/unlabelled_news_train_reformatted.csv', index=False)
# val.to_csv(f'texts/unlabelled_news_val_reformatted.csv', index=False)

# ###Save the text data
# unlabelled_data.to_csv(f'texts/unlabelled_news_reformatted.csv', index=False)

# print("Total image-text pairs in pretraining CLIP", len(unlabelled_data))


###Reformat the labelled data

###The labelled data is in a json from label studio 

##Load the json
# train_data_path="/mnt/data01/clippings_general/texts/emily_news_captions_3.json"
train_data_list=["/mnt/data01/clippings_general/texts/emily_news_captions_3.json","/mnt/data01/clippings_general/texts/emily_news_captions_2.json","/mnt/data01/clippings_general/texts/emily_newspaper_labels_0603.json"]
train_data_list=["/mnt/data01/clippings_general/texts/emily_news_captions_2.json"]

###Concat the jsons
data = []
for path in train_data_list:
    with open(path) as f:
        data += json.load(f)



###Make two lists of pairs of image-captions and results. (image1,caption1,image2,caption2,result)
image1 = []
caption1 = []
image2 = []
caption2 = []
result = []

for i in range(len(data)):
    # print(i)
    image1.append(data[i]['data']['id1'])
    caption1.append(data[i]['data']['caption1'])
    image2.append(data[i]['data']['id2'])
    caption2.append(data[i]['data']['caption2'])

    ##Get result if it exists
    if len(data[i]['annotations'][0]['result'])>0:
        result.append(data[i]['annotations'][0]['result'][0]['value']['choices'][0])
    else:
        result.append('Drop')

###Make a dataframe
train_data = pd.DataFrame({'image1':image1, 'caption1':caption1, 'image2':image2, 'caption2':caption2, 'result':result})

###ADditonnaly, build a df with just the image paths and captions - stack the image1 and image2 columns and the caption1 and caption2 columns
image_paths = pd.concat([train_data['image1'], train_data['image2']],axis=0)
captions = pd.concat([train_data['caption1'], train_data['caption2']],axis=0)
image_captions = pd.DataFrame({'image_path':image_paths, 'caption':captions})


###Now, replace "Different" with 0 and "Same" with 1
train_data['result'] = train_data['result'].replace({'Different':0, 'Same':1,'Drop':np.nan})

##DRop if result is nan
train_data = train_data.dropna()
image_1= train_data['image1'].tolist()
image_2= train_data['image2'].tolist()
labels= train_data['result'].tolist()

###Build a graph from the data. Nodes are connected if they have result=1
edges_list = []
for i in range(len(image_1)):
    if labels[i] == 1:
        edges_list.append([image_1[i], image_2[i]])

cluster_dict = clusters_from_edges(edges_list)

print(cluster_dict)

##Check if there are any clusters with more than 2 image
for cluster in cluster_dict:
    if len(cluster_dict[cluster])>2:
        print(cluster_dict[cluster])

##Make a dataframe with members of each cluster as a row and a column with the cluster id
cluster_df = pd.DataFrame(columns=['image_path', 'cluster_id'])
for cluster in cluster_dict:
    for image in cluster_dict[cluster]:
        cluster_df = cluster_df.append({'image_path':image, 'cluster_id':cluster},ignore_index=True)

###Add singletons to the clusters
all_images_in_clusters = [item for sublist in cluster_dict.values() for item in sublist]
singletons_1 = [image for image in image_1 if image not in all_images_in_clusters]
singletons_2 = [image for image in image_2 if image not in all_images_in_clusters]
singletons = list(set(singletons_1 + singletons_2))


for i,image in enumerate(singletons):
    cluster_df = cluster_df.append({'image_path':image, 'cluster_id':-(i)},ignore_index=True)


##Drop duplicates
cluster_df = cluster_df.drop_duplicates()

###Now merge the cluster_df with the image_captions df
connected_components_df = pd.merge(cluster_df, image_captions, on='image_path', how='left')

###Drop duplicates
connected_components_df = connected_components_df.drop_duplicates()

###Rename the columns cluster_id to label, caption to text
connected_components_df = connected_components_df.rename(columns={'cluster_id':'label', 'caption':'text'})




###Add root to the image path

connected_components_df['image_path'] = '/mnt/data02/captions/pulled_crops_quicker_all/' + connected_components_df['image_path'] + '.png'

##Save the text data
connected_components_df.to_csv(f'/mnt/data01/clippings_general/texts/labelled_news_reformatted.csv', index=False)

##Before dropping duplicates, check if there are any duplicates
print("Number of duplicates", len(connected_components_df[connected_components_df.duplicated()]))

print("length before dropping duplicates", len(connected_components_df))

###Drop duplicates
connected_components_df = connected_components_df.drop_duplicates()
connected_components_df.to_csv(f'/mnt/data01/clippings_general/texts/labelled_news_reformatted.csv', index=False)

##Reorder the columns
connected_components_df = connected_components_df[['image_path', 'text', 'label']]

print("length after dropping duplicates", len(connected_components_df))

##Split into train and val by using labels. Sample 20% of the labels for val
train_labels, val_labels = train_test_split(connected_components_df['label'].unique(), test_size=0.2, random_state=42)

print(train_labels[:10])
##Split the data
train = connected_components_df[connected_components_df['label'].isin(train_labels)]
print("Number of train images", len(train))
val = connected_components_df[connected_components_df['label'].isin(val_labels)]
print("Number of val images", len(val))





##Save the text data
train.to_csv(f'/mnt/data01/clippings_general/texts/labelled_news_train_reformatted.csv', index=False)
val.to_csv(f'/mnt/data01/clippings_general/texts/labelled_news_val_reformatted.csv', index=False)


##Save a version without singleton images
train = train[train['label']>=0]
val = val[val['label']>=0]

##Save the text data
train.to_csv(f'/mnt/data01/clippings_general/texts/labelled_news_train_reformatted_no_singletons.csv', index=False)
val.to_csv(f'/mnt/data01/clippings_general/texts/labelled_news_val_reformatted_no_singletons.csv', index=False)

# print("Total image-text pairs in pretraining CLIP", len(connected_components_df))

# ###Reformat the eval data

# ###Eval data csv
# eval_data_path="/mnt/data02/captions/labels/test_day.csv"

# eval_data = pd.read_csv(eval_data_path)

# ####Add root to the image path

# eval_data['image_path'] = '/mnt/data02/captions/test_day_pulled_crops/' + eval_data['image_id'] + '.png'

# ##Drop image_id
# eval_data = eval_data.drop(columns=['image_id'])

# ##Remove the unnamed column
# eval_data = eval_data.drop(columns=['Unnamed: 0'])

# ##Rename caption as text and cluster_label as label
# eval_data = eval_data.rename(columns={'caption':'text', 'cluster_label':'label'})

# ##Reorder and keep only image_path, text and label
# eval_data = eval_data[['image_path', 'text', 'label']]



# ##Save the text data
# eval_data.to_csv(f'/mnt/data01/clippings_general/texts/labelled_news_eval_reformatted.csv', index=False)

# print("Total image-text pairs in eval CLIP", len(eval_data))

