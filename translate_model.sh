#!/bin/sh
#SBATCH -A uppmax2024-2-13
#SBATCH -p core -n 4
#SBATCH -M snowy
#SBATCH -t 2:00:00  # Reduced time as translation usually takes less time than training
#SBATCH -J onmt_translate_acholi
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=my_email*@gmail.com

# Set CUDA visible devices
CUDA_VISIBLE_DEVICES=0

# Directory settings
PROJECT_DIR=/proj/uppmax2024-2-13/private/acholi_mt24/project-acholi-mt24
DATA_DIR=$PROJECT_DIR/onmt_data
MODEL_DIR=$PROJECT_DIR/onmt_data/onmt_model
OUTPUT_DIR=$PROJECT_DIR/translations
mkdir -p $OUTPUT_DIR

# Model path - adjust the step number as needed
MODEL_PATH=$MODEL_DIR/model_step_100000.pt  # Change this to your best model checkpoint

# Test data paths
TEST_SRC=$DATA_DIR/test.bpe.ach  # Your BPE-encoded test source file
OUTPUT_FILE=$OUTPUT_DIR/test_predictions.txt
GOLD_FILE=$DATA_DIR/test.en  # Your reference file (if available)

# Log directory
LOG_DIR=$OUTPUT_DIR/logs
mkdir -p $LOG_DIR

# Check if GPU is available
nvidia-smi
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo "Starting translation with OpenNMT-py"

# Translation command
onmt_translate \
    -model $MODEL_PATH \
    -src $TEST_SRC \
    -output $OUTPUT_FILE \
    -gpu 0 \
    -batch_size 32 \
    -beam_size 5 \
    -replace_unk \
    2>&1 | tee $LOG_DIR/translation.log

# If you have reference translations, calculate BLEU score
if [ -f "$GOLD_FILE" ]; then
    echo "Calculating BLEU score..."
    # First, remove BPE
    sed -i 's/@@ //g' $OUTPUT_FILE
    
    # Calculate BLEU using sacrebleu
    cat $OUTPUT_FILE | sacrebleu $GOLD_FILE --width 2 > $LOG_DIR/bleu_score.txt
    echo "BLEU score saved to $LOG_DIR/bleu_score.txt"
fi

echo "Translation completed. Output saved to $OUTPUT_FILE"