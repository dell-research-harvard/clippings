# Replicate all results in the main table


# Rule based Jaccard Similarity -> SLINK -> ARI (fill in paths for val and test data)
python scripts/jaccard_sim.py 

#  SelfSup Visual Linking
python infer_clippings.py --im_wt 1 --specified_thresh 0.1175  --split_test_for_eval --checkpoint_path path/to/checkpoint.pt 

# SelfSup Language Linking
python infer_clippings.py --im_wt 0 --specified_thresh 0.08  --split_test_for_eval --checkpoint_path path/to/checkpoint.pt 


# SelfSup Multimodal Linking
python infer_clippings.py --im_wt 0.636  --specified_thresh 0.13  --split_test_for_eval --checkpoint_path path/to/checkpoint.pt 


# Supervised Visual Linking
python infer_clippings.py --im_wt 1 --specified_thresh 0.251  --split_test_for_eval --checkpoint_path path/to/checkpoint.pt 

# Supervised Language Linking
python infer_clippings.py --im_wt 0 --specified_thresh 0.164  --split_test_for_eval --checkpoint_path path/to/checkpoint.pt 

# Supervised Multimodal Linking
python infer_clippings.py --im_wt 0.76 --specified_thresh 0.227  --split_test_for_eval --checkpoint_path path/to/checkpoint.pt 
