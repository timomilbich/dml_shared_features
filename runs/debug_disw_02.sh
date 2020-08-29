export GPU_TRAINING=$1
echo "gpu id is ${GPU_TRAINING}"
export WANDB_KEY=8388187e7c47589ca2875e4007015c7536aede7f
echo "wandb key is ${WANDB_KEY}"
####################################
####################################

### DEBUG DISW
# ... margin loss
python main_shared_features.py --savename cub200_disw_1800 --tau 55 80 --disw 1800 --dataset cub200 --n_epochs 150 --log_online --project dml_shared_features --group debug_disw --seed 23 --gpu ${GPU_TRAINING} --wandb_key ${WANDB_KEY}
python main_shared_features.py --savename cub200_disw_2000 --tau 55 80 --disw 2000 --dataset cub200 --n_epochs 150 --log_online --project dml_shared_features --group debug_disw --seed 23 --gpu ${GPU_TRAINING} --wandb_key ${WANDB_KEY}
python main_shared_features.py --savename cub200_disw_2200 --tau 55 80 --disw 2200 --dataset cub200 --n_epochs 150 --log_online --project dml_shared_features --group debug_disw --seed 23 --gpu ${GPU_TRAINING} --wandb_key ${WANDB_KEY}
python main_shared_features.py --savename cub200_disw_2400 --tau 55 80 --disw 2400 --dataset cub200 --n_epochs 150 --log_online --project dml_shared_features --group debug_disw --seed 23 --gpu ${GPU_TRAINING} --wandb_key ${WANDB_KEY}
python main_shared_features.py --savename cub200_disw_2600 --tau 55 80 --disw 2600 --dataset cub200 --n_epochs 150 --log_online --project dml_shared_features --group debug_disw --seed 23 --gpu ${GPU_TRAINING} --wandb_key ${WANDB_KEY}
