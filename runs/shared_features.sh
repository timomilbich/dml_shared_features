export GPU_TRAINING=$1
echo "gpu id is ${GPU_TRAINING}"
export WANDB_KEY=8388187e7c47589ca2875e4007015c7536aede7f
echo "wandb key is ${WANDB_KEY}"
####################################
####################################

### BASELINE CHECKS
# ... margin loss
python main_shared_features_cars196.py --savename test_wandb_cars196 --dataset cars196 --n_epochs 150 --log_online --project dml_shared_features --group shared_features_baseline --seed 23 --gpu ${GPU_TRAINING} --wandb_key ${WANDB_KEY}