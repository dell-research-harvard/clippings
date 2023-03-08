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
    print(i)
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
image_paths = pd.concat([train_data['image1'], train_data['image2']])
captions = pd.concat([train_data['caption1'], train_data['caption2']])
image_captions = pd.DataFrame({'image_path':image_paths, 'caption':captions})


###Now, replace "Different" with 0 and "Same" with 1
train_data['result'] = train_data['result'].replace({'Different':0, 'Same':1,'Drop':np.nan})


###Build a graph from the data. Nodes are connected if they have result=1

connected_pairs = train_data[train_data['result']==1]

###Make a graph

G=nx.from_pandas_edgelist(connected_pairs, 'image1', 'image2')

l=list(nx.connected_components(G))

L=[dict.fromkeys(y,x) for x, y in enumerate(l)]

d={k: v for d in L for k, v in d.items()}

print(d)

###Make a two column df with image path and label
connected_components_df = pd.DataFrame.from_dict(d, orient='index').reset_index()

print((connected_components_df))

###Rename the columns
connected_components_df.columns = ['image_path', 'label']


print((connected_components_df))
print(len(connected_components_df), "connected components")


###Add singletons
singletons = train_data[(train_data['result']==0) & (~train_data['image1'].isin(connected_components_df.image_path)) &  (~train_data['image2'].isin(connected_components_df.image_path))]
print("Number of singletons", len(singletons))

###Now split the singletons into two columns
singletons = pd.concat([singletons['image1'], singletons['image2']], axis=0).reset_index(drop=True)


###Add the label column - -index would be the label

###Add a label to the singletons
single_ton_labels=[-i-1 for i in range(len(singletons))]
singletons = pd.DataFrame({'image_path':singletons, 'label':single_ton_labels})


print(singletons.head(10))


print("LEngth before singletons", len(connected_components_df))
####Add singletons to the connected components
connected_components_df = pd.concat([connected_components_df, singletons], axis=0)

print("LEngth after singletons", len(connected_components_df))
###merge the captions to the extended connected components df
connected_components_df = pd.merge(connected_components_df, image_captions, on='image_path')



###Rename caption as text and reorder columns
connected_components_df = connected_components_df.rename(columns={'caption':'text'})
connected_components_df = connected_components_df[['image_path', 'text', 'label']]

##Print len unique labels
print("Number of unique labels", len(connected_components_df['label'].unique()))

###Add root to the image path

connected_components_df['image_path'] = '/mnt/data02/captions/train_day_pulled_crops_quicker/' + connected_components_df['image_path'] + '.png'

##Save the text data
connected_components_df.to_csv(f'/mnt/data01/clippings_general/texts/labelled_news_reformatted.csv', index=False)

##Before dropping duplicates, check if there are any duplicates
print("Number of duplicates", len(connected_components_df[connected_components_df.duplicated()]))

###Drop duplicates
eval_data = connected_components_df.drop_duplicates()

##Split into train and val by using labels. Sample 20% of the labels for val
train, val = train_test_split(connected_components_df['label'].unique(), test_size=0.2, random_state=42)

##Split the data
train = connected_components_df[connected_components_df['label'].isin(train)]
print("Number of train images", len(train))
val = connected_components_df[connected_components_df['label'].isin(val)]
print("Number of val images", len(val))



##Save the text data
train.to_csv(f'/mnt/data01/clippings_general/texts/labelled_news_train_reformatted.csv', index=False)
val.to_csv(f'/mnt/data01/clippings_general/texts/labelled_news_val_reformatted.csv', index=False)


print("Total image-text pairs in pretraining CLIP", len(connected_components_df))

###Reformat the eval data

###Eval data csv
eval_data_path="/mnt/data02/captions/labels/test_day.csv"

eval_data = pd.read_csv(eval_data_path)

####Add root to the image path

eval_data['image_path'] = '/mnt/data02/captions/test_day_pulled_crops/' + eval_data['image_id'] + '.png'

##Drop image_id
eval_data = eval_data.drop(columns=['image_id'])

##Remove the unnamed column
eval_data = eval_data.drop(columns=['Unnamed: 0'])

##Rename caption as text and cluster_label as label
eval_data = eval_data.rename(columns={'caption':'text', 'cluster_label':'label'})

##Reorder and keep only image_path, text and label
eval_data = eval_data[['image_path', 'text', 'label']]



##Save the text data
eval_data.to_csv(f'/mnt/data01/clippings_general/texts/labelled_news_eval_reformatted.csv', index=False)

print("Total image-text pairs in eval CLIP", len(eval_data))

