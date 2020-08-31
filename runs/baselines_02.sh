export GPU_TRAINING=$1
echo "gpu id is ${GPU_TRAINING}"
export WANDB_KEY=8388187e7c47589ca2875e4007015c7536aede7f
echo "wandb key is ${WANDB_KEY}"
####################################
####################################

### BASELINE CHECKS
# ... multisimilarity loss
python main_baselines.py --savename cub200_r50_128_multisimilarity_noSchedule --dataset cub200 --tau 999 --class_loss multisimilarityloss --n_epochs 150 --log_online --project dml_shared_features --group baselines --seed 23 --gpu ${GPU_TRAINING} --wandb_key ${WANDB_KEY}

