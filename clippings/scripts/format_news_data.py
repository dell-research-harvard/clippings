###Reoganise food 101 data

import os
import pandas as pd
import numpy as np
###Import test_train_split
from sklearn.model_selection import train_test_split


###Load all unlabelled data

###Reformat the newspaper images data
###Load the directory containig the csvs
unlabelled_dir = '/mnt/data02/captions/unlabelled_pairs/'
##Load all the csvs
unlabelled_data = pd.concat([pd.read_csv(unlabelled_dir + file) for file in os.listdir(unlabelled_dir)])
##Remove the index column
unlabelled_data = unlabelled_data.drop(columns=['Unnamed: 0'])
###Add a column with label=999
unlabelled_data['label'] = 999
##Rename the columns
unlabelled_data.columns = ['image_path', 'text', 'label']

##Add base path to image path
unlabelled_data['image_path'] = '/mnt/data02/captions/pulled_crops/' + unlabelled_data['image_path'] + '.png'

##Split in val and train
train, val = train_test_split(unlabelled_data, test_size=0.2, random_state=42)

##Save the text data
train.to_csv(f'texts/unlabelled_news_train_reformatted.csv', index=False)
val.to_csv(f'texts/unlabelled_news_val_reformatted.csv', index=False)

###Save the text data
unlabelled_data.to_csv(f'texts/unlabelled_news_reformatted.csv', index=False)

print("Total image-text pairs in pretraining CLIP", len(unlabelled_data))


###Reformat the labelled data






