# clippings

This repo is for CLIPPINGs as a clean and simple general purpose solution for multi-modal metric learning.
This readme file is for the associated paper's ICCV submission. Given the size limit of the submission, we only show a sample of the data. 

## File tree


- sample_data : Contains sample data
    -  images
    -  image_captions.csv

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
Use relevant hyperparameters from Table X in the supplementary material file

Pretrain the base CLIP model (example)

```
python train_clippings.py --clip_lr 5e-6 --train_data_type newspapers_unlabelled --wandb_name clip_pretrain_unlabelled_m1_newspapers_cc --training_type pretrain

```

Train the CLIPPINGS model (example)

```
python train_clippings.py --clip_lr 5e-9 --train_data_type newspapers_labelled --wandb_name bienc_clip_nopretrain_labelled_m3_v3_newspapers_nosingle --m 3 --training_type train_bienc --im_wt 0.5 --k 3 --supcon_temp 0.1 --train_hardneg --loss_type supcon --checkpoint /path/to/pretrained/clip.pt
```

### Evaluation



``` 
python infer_clippings.py --im_wt {a} --specified_thresh {b}  --split_test_for_eval --checkpoint_path path/to/checkpoint.pt 

```
Use a and b from the the supplementary material table X to replicate the relevant result given model weights. Addtionally, use args --opt_im_wt to optmise the weight of image embeddings in the final CLIPPINGS embedding and remove the --specified_thresh argument to re-optimise the threshold given the val set. 
