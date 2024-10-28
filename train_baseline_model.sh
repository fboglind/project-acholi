#!/bin/sh
#SBATCH -A uppmax2024-2-13
#SBATCH -p core -n 4 
#SBATCH -M snowy 
## # SBATCH -t 24:00:00 
#SBATCH -t 00:15:00
#SBATCH -J onmt_acholi_en
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=fredrik.boglind@gmail.com 
#SBATCH --qos=short

# # #SBATCH --qos=gpu
## Uncomment the above line if you're running under project 2002-2-2
## For testing, you can also use "--qos=short", for a maximum of 15 minutes

# Set CUDA visible devices
CUDA_VISIBLE_DEVICES=0

# Directory settings
DATA_DIR=/proj/uppmax2024-2-13/private/acholi_mt24/project-acholi-mt24/onmt_data
SAVE_DIR=/proj/uppmax2024-2-13/private/acholi_mt24/project-acholi-mt24/onmt_data/onmt_model
mkdir -p $SAVE_DIR

# Configuration file for training
CONFIG_FILE=$DATA_DIR/train_config.yaml

# Ensure the config file exists
if [ ! -f $CONFIG_FILE ]; then
  echo "Configuration file not found: $CONFIG_FILE"
  exit 1
fi

echo "Starting training with OpenNMT-py"

onmt_train -config $CONFIG_FILE \
  --save_model $SAVE_DIR/model \
  --world_size 1 --gpu_ranks 0 \
  --train_steps 50000 \
  --valid_steps 1000 \
  --save_checkpoint_steps 5000 \
  --keep_checkpoint 5 \
  --early_stopping 5 \
  --report_every 100

echo "Training completed"
