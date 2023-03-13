# clippings

This repo is for CLIPPINGs as a clean and simple general purpose solution for multi-modal metric learning.
This readme file is for the associated paper's ICCV submission. Given the size limit of the submission, we only show a sample of the data. 

## File tree


- sample_data : Contains sample data
    -  newscaptions_dup

- datasets
    - data_loaders.py : Contains the pytorch datasets, dataloaders and miners neccesary for training

- models
    - encoders.py: Contains a class defining an MLP which was a part of the experiments instead of mean-pooling (deprecated later, but the training script retains functionality)

- scripts
    - format_news_data_v2.py : To prepare the labelled and unlabelled data for newspaper image-captions (for both train and test)

- utils 
    - datasets_utils.py : Contains the main torchvision transformations needed to transform the images before feeding them into the model
    - gen_synthetic_segments : Some data augmentation functions to prepare random augmentations (the codebase retains the functionality but no augmentations are used in the final results)

- train_clippings.py : Script that supports both language-image pretraining of an underlying clip model as well as the main function to train "CLIPPINGS"

- infer_clippings.py : Run inference to embed image-text pairs given model weights, perform SLINK, find the optimum threshold using the val set and present ARI for the test data

- requirements.yaml : The conda environment containing all dependencies

## Code usage
This section provides the commands (with relevant arguments) to replicate the results in the main paper. 

### Train



### Evaluation



``` 
python infer_clippings.py --im_wt {a} --use  --split_test_for_eval --checkpoint_path /mnt/data01/clippings_general/models/clip_imwt_5bienc_clip_nopretrain_labelled_m3_v3_newspapers_nosingle.pt  

```


