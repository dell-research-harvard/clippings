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
###Save the text data
unlabelled_data.to_csv(f'texts/unlabelled_news_reformatted.csv', index=False)


###Reformat the labelled data





