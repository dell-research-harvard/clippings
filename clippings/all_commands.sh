
###All commands must be run from the root of the repo

CUDA_VISIBLE_DEVICES=0 python clippings/continue_pretrain_clip.py --clip_lr 5e-6 --train_data_type newspapers_labelled --wandb_name bienc_clip_pretrain_labelled_m3_v3_newspapers --m 3 --training_type train_bienc --im_wt 0.5 --k 3 --supcon_temp 0.1 --train_hardneg --checkpoint "/mnt/data01/clippings_general/models/clip_pretrain_unlabelled_m1_newspapers_cc.pt"


CUDA_VISIBLE_DEVICES=1 python clippings/continue_pretrain_clip.py --clip_lr 5e-9 --train_data_type newspapers_labelled --wandb_name bienc_clip_pretrain_labelled_m3_v3_newspapers_lowlr --m 3 --training_type train_bienc --im_wt 0.5 --k 3 --supcon_temp 0.1 --train_hardneg --checkpoint "/mnt/data01/clippings_general/models/clip_pretrain_unlabelled_m1_newspapers_cc.pt"


CUDA_VISIBLE_DEVICES=1 python clippings/continue_pretrain_clip.py --clip_lr 5e-9 --train_data_type newspapers_labelled --wandb_name bienc_clip_pretrain_labelled_m3_v3_newspapers_nosingle --m 3 --training_type train_bienc --im_wt 0.5 --k 3 --supcon_temp 0.1 --train_hardneg --checkpoint "/mnt/data01/clippings_general/models/clip_pretrain_unlabelled_m1_newspapers_cc.pt"



####Contrastive loss
CUDA_VISIBLE_DEVICES=0 python clippings/continue_pretrain_clip.py --clip_lr 5e-5 --train_data_type newspapers_labelled --wandb_name bienc_clip_pretrain_labelled_m3_v3_newspapers_contrastive --m 3 --training_type train_bienc --im_wt 0.5 --k 3 --supcon_temp 0.1 --train_hardneg --checkpoint "/mnt/data01/clippings_general/models/clip_pretrain_unlabelled_m1_newspapers_cc.pt" --loss_type "contrastive" --contrastive_loss_pos_margin 0 --contrastive_loss_neg_margin 0.5


###Inference ARI
CUDA_VISIBLE_DEVICES=1 python clippings/inference_clip.py --im_wt 0.5 --checkpoint_path /mnt/data01/clippings_general/models/clip_imwt_5bienc_clip_pretrain_labelled_m3_v3_newspapers_contrastive.pt
