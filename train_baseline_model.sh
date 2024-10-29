#!/bin/sh
#SBATCH -A uppmax2024-2-13
#SBATCH -p core -n 4 
#SBATCH -M snowy 
#SBATCH -t 24:00:00 
#SBATCH -J onmt_acholi_en2
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your*@email.com

## To run: sbatch train_baseline_model.sh (NOT bash!)

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

## Check if GPU is used
nvidia-smi
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

export PYTORCH_ENABLE_MPS_FALLBACK=1
export ONMT_LOG_LEVEL=WARNING ## Reduces verbosity
LOG_DIR=$SAVE_DIR/logs
mkdir -p $LOG_DIR

echo "Starting training with OpenNMT-py"

onmt_train -config $CONFIG_FILE \
  --save_model $SAVE_DIR/model \
  --world_size 1 --gpu_ranks 0 \
  --keep_checkpoint 5 2>&1 | grep -v "Weighted corpora loaded" | tee $LOG_DIR/training.log

## Get summary
grep -E "Step |Validation|BLEU" $LOG_DIR/training.log > $LOG_DIR/training_summary.log
echo "Training completed"
