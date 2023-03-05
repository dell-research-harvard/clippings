###Reoganise food 101 data

import os
import pandas as pd
import numpy as np



def format_food101_data(split='train'):
    ###Load the text file with the labels
    text_data = pd.read_csv(f'texts/{split}_titles.csv', header=None)

    ###Add column names
    text_data.columns = ['image_path', 'text', 'label']

    ###Append base paths to image paths
    text_data['image_path'] = '/mnt/data01/clippings_general/images/' + text_data['label'] +"/" + text_data['image_path']

    ###Save the text data
    text_data.to_csv(f'texts/{split}_titles_reformatted.csv', index=False)



##Run it for train and test
format_food101_data(split='train')
format_food101_data(split='test')