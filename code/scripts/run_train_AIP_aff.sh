python train_AIP_aff.py \
  --exp_suffix model_AIP_aff_para \
  --train_data_list ../data/train_push_data_dir \
  --val_data_list ../data/val_push_data_dir \
  --critic_dir ../logs/model_AAP_para \
  --critic_epoch 40 \
  --AIP_dir ../logs/model_AIP_para \
  --AIP_epoch 40