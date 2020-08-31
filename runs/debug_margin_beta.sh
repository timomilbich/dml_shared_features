export GPU_TRAINING=$1
echo "gpu id is ${GPU_TRAINING}"
export WANDB_KEY=8388187e7c47589ca2875e4007015c7536aede7f
echo "wandb key is ${WANDB_KEY}"
####################################
####################################

### NO SCHEDULING
# ... margin loss (intra_beta=0.6 | class_beta = 0.6)
python main_shared_features.py --savename cars196_disw_900_betas0.6 --tau 999 --class_beta 0.6 --intra_beta 0.6 --disw 900 --dataset cars196 --n_epochs 150 --log_online --project dml_shared_features --group noSchedule --seed 23 --gpu ${GPU_TRAINING} --wandb_key ${WANDB_KEY}
python main_shared_features.py --savename cub200_disw_2400_betas0.6 --tau 999 --class_beta 0.6 --intra_beta 0.6 --disw 2400 --dataset cub200 --n_epochs 150 --log_online --project dml_shared_features --group noSchedule --seed 23 --gpu ${GPU_TRAINING} --wandb_key ${WANDB_KEY}