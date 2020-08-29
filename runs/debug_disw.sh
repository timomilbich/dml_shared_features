export GPU_TRAINING=$1
echo "gpu id is ${GPU_TRAINING}"
export WANDB_KEY=8388187e7c47589ca2875e4007015c7536aede7f
echo "wandb key is ${WANDB_KEY}"
####################################
####################################

### DEBUG DISW
# ... margin loss
python main_shared_features_cars196.py --savename cars196_disw_300 --disw 300 --dataset cars196 --n_epochs 150 --log_online --project dml_shared_features --group debug_disw --seed 23 --gpu ${GPU_TRAINING} --wandb_key ${WANDB_KEY}
python main_shared_features_cars196.py --savename cars196_disw_700 --disw 700 --dataset cars196 --n_epochs 150 --log_online --project dml_shared_features --group debug_disw --seed 23 --gpu ${GPU_TRAINING} --wandb_key ${WANDB_KEY}
python main_shared_features_cars196.py --savename cars196_disw_900 --disw 900 --dataset cars196 --n_epochs 150 --log_online --project dml_shared_features --group debug_disw --seed 23 --gpu ${GPU_TRAINING} --wandb_key ${WANDB_KEY}
python main_shared_features_cars196.py --savename cars196_disw_1100 --disw 1100 --dataset cars196 --n_epochs 150 --log_online --project dml_shared_features --group debug_disw --seed 23 --gpu ${GPU_TRAINING} --wandb_key ${WANDB_KEY}
