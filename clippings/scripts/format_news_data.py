###Reoganise food 101 data

import os
import pandas as pd
import numpy as np
###Import test_train_split
from sklearn.model_selection import train_test_split
import json
import networkx as nx

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

###Load the json
train_data_path="/mnt/data01/clippings_general/texts/emily_newspaper_labels_0603.json"

with open(train_data_path) as f:
    data = json.load(f)


###Make two lists of pairs of image-captions and results. (image1,caption1,image2,caption2,result)
image1 = []
caption1 = []
image2 = []
caption2 = []
result = []

for i in range(len(data)):
    print(i)
    image1.append(data[i]['data']['id1'])
    caption1.append(data[i]['data']['caption1'])
    image2.append(data[i]['data']['image2'])
    caption2.append(data[i]['data']['caption2'])

    ##Get result if it exists
    if len(data[i]['annotations'][0]['result'])>0:
        result.append(data[i]['annotations'][0]['result'][0]['value']['choices'][0])
    else:
        result.append('Drop')

###Make a dataframe
train_data = pd.DataFrame({'image1':image1, 'caption1':caption1, 'image2':image2, 'caption2':caption2, 'result':result})

###ADditonnaly, build a df with just the image paths and captions - stack the image1 and image2 columns and the caption1 and caption2 columns
image_paths = pd.concat([train_data['image1'], train_data['image2']])
captions = pd.concat([train_data['caption1'], train_data['caption2']])
image_captions = pd.DataFrame({'image_path':image_paths, 'caption':captions})


###Now, replace "Different" with 0 and "Same" with 1
train_data['result'] = train_data['result'].replace({'Different':0, 'Same':1,'Drop':np.nan})


###Build a graph from the data. Nodes are connected if they have result=1

connected_pairs = train_data[train_data['result']==1]

###Make a graph
G = nx.Graph()

###Add all the nodes
G.add_nodes_from(train_data['image1'])
G.add_nodes_from(train_data['image2'])

###Add all the edges
for i in range(len(connected_pairs)):
    G.add_edge(connected_pairs['image1'].iloc[i], connected_pairs['image2'].iloc[i])

###Find the connected components
connected_components = list(nx.connected_components(G))

###Make a dictionary with the connected components
connected_components_dict = {}
for i in range(len(connected_components)):
    connected_components_dict[i] = list(connected_components[i])

###Make a dataframe with the connected components
connected_components_df = pd.DataFrame.from_dict(connected_components_dict, orient='index')

###Add the label column
connected_components_df['label'] = connected_components_df.index


print((connected_components_df.head(5)))

##Rename the 0 column to image_path
connected_components_df = connected_components_df.rename(columns={0:'image_path'})
###Add singletons
singletons = train_data[train_data['result']==0]

###Add the label column - start from the last label
singletons['label'] = connected_components_df.index.max() + 1

####Add singletons to the connected components
connected_components_df = pd.concat([connected_components_df, singletons])

###merge the captions to the extended connected components df
connected_components_df = pd.merge(connected_components_df, image_captions, on='image_path')







