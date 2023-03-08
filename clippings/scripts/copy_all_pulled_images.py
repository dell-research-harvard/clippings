###Copy all crops to new folder

import os
import shutil
import glob

###Images to copy 

new_folder='/mnt/data02/captions/pulled_crops_quicker_all/'

list_of_images = glob.glob('/mnt/data02/captions/train_day_pulled_crops_quicker_3/*.png')

print("Number of images", len(list_of_images))

# ##Create new folder if it doesn't exist
# if not os.path.exists(new_folder):
#     os.makedirs(new_folder)



for image in list_of_images:
    shutil.copy(image, new_folder)
    