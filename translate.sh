#!/bin/sh
#SBATCH -A uppmax2024-2-13
#SBATCH -p core -n 4
#SBATCH -M snowy
#SBATCH -t 4:00:00
#SBATCH -J onmt_translate_acholi
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=fredrik.boglind@gmail.com

# Set CUDA visible devices
CUDA_VISIBLE_DEVICES=0

# Directories
PROJECT_DIR=/home/oslj9489/private/project-acholi-mt24
MODEL_PATH=$PROJECT_DIR/onmt_data/onmt_model/model_step_6500.pt  # your latest checkpoint
TEST_SRC=$PROJECT_DIR/onmt_data/test.bpe.ach
OUTPUT_DIR=$PROJECT_DIR/onmt_data/test_translations
mkdir -p $OUTPUT_DIR

# Translate
onmt_translate \
    -model $MODEL_PATH \
    -src $TEST_SRC \
    -output $OUTPUT_DIR/predictions.txt \
    -gpu 0 \
    -batch_size 32 \
    -beam_size 5 \
    -replace_unk

# Remove BPE
sed -i 's/@@ //g' $OUTPUT_DIR/predictions.txt

# Calculate BLEU if you have references
sacrebleu $PROJECT_DIR/processed_data_moses/salt.test.tk.lc.eng \
    < $OUTPUT_DIR/predictions.txt > $OUTPUT_DIR/bleu_score.txt