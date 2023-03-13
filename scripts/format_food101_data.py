###Reoganise food 101 data

import os
import pandas as pd
import numpy as np
###Import test_train_split
from sklearn.model_selection import train_test_split



def format_food101_data(split='train'):
    ###Load the text file with the labels
    text_data = pd.read_csv(f'texts/{split}_titles.csv', header=None)

    ###Add column names
    text_data.columns = ['image_path', 'text', 'label']

    ###Append base paths to image paths
    text_data['image_path'] = '/mnt/data01/clippings_general/images/' +split +"/" + text_data['label'] +"/" + text_data['image_path']


    ##Split into train and val if split=="train"
    if split=="train":
        ###Split into train and val
        train, val = train_test_split(text_data, test_size=0.2, random_state=42)

        ###Save the text data
        train.to_csv(f'texts/train_titles_reformatted.csv', index=False)
        val.to_csv(f'texts/val_titles_reformatted.csv', index=False)

    else:
        ###Save the text data
        text_data.to_csv(f'texts/{split}_titles_reformatted.csv', index=False)





##Run it for train and test
format_food101_data(split='train')
format_food101_data(split='test')