#!/bin/bash

# Base path for Moses tools - updated to use environment's Moses installation
MOSES_PATH="$HOME/envs/acholi_mt_env/tools/mosesdecoder"

# Input and output directories
DATA_DIR="data"
OUTPUT_DIR="processed_data_moses"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

SPLITS=("train" "dev" "test")

# Function to check if a command was successful
check_status() {
    if [ $? -eq 0 ]; then
        echo "✓ $1 successful"
    else
        echo "✗ Error during $1"
        exit 1
    fi
}

# Function to process a single split
process_split() {
    local split=$1
    echo "Processing $split set..."

    # Input files
    ACH_INPUT="$DATA_DIR/salt.$split.ach"
    ENG_INPUT="$DATA_DIR/salt.$split.en"

    # Check if input files exist
    if [ ! -f "$ACH_INPUT" ] || [ ! -f "$ENG_INPUT" ]; then
        echo "Error: Input files for $split set not found"
        exit 1
    fi

    # Tokenization
    echo "Tokenizing $split files..."
    perl $MOSES_PATH/scripts/tokenizer/tokenizer.perl -l en < $ENG_INPUT > $OUTPUT_DIR/salt.$split.tk.eng
    check_status "English tokenization for $split"
    perl $MOSES_PATH/scripts/tokenizer/tokenizer.perl -l en < $ACH_INPUT > $OUTPUT_DIR/salt.$split.tk.ach
    check_status "Acholi tokenization for $split"

    # Lowercasing
    echo "Lowercasing $split files..."
    perl $MOSES_PATH/scripts/tokenizer/lowercase.perl < $OUTPUT_DIR/salt.$split.tk.eng > $OUTPUT_DIR/salt.$split.tk.lc.eng
    check_status "English lowercasing for $split"
    perl $MOSES_PATH/scripts/tokenizer/lowercase.perl < $OUTPUT_DIR/salt.$split.tk.ach > $OUTPUT_DIR/salt.$split.tk.lc.ach
    check_status "Acholi lowercasing for $split"

    # Truecasing (commented out as in original script)
    # echo "Truecasing $split files..."
    # perl $MOSES_PATH/scripts/recaser/truecase.perl < $OUTPUT_DIR/salt.$split.tk.eng > $OUTPUT_DIR/salt.$split.tk.tc.eng
    # perl $MOSES_PATH/scripts/recaser/truecase.perl < $OUTPUT_DIR/salt.$split.tk.ach > $OUTPUT_DIR/salt.$split.tk.tc.ach

    # Cleaning (only for train set)
    if [ "$split" == "train" ]; then
        echo "Cleaning $split corpus..."
        perl $MOSES_PATH/scripts/training/clean-corpus-n.perl \
            $OUTPUT_DIR/salt.$split.tk.lc ach eng \
            $OUTPUT_DIR/salt.$split.tk.lc.clean 1 40
        check_status "Cleaning for $split"
    fi
}

# Make sure we're in the right environment
if [[ ! "$VIRTUAL_ENV" =~ "acholi_mt_env" ]]; then
    echo "Error: Please activate the acholi_mt_env first:"
    echo "source ~/envs/activate_acholi_mt_env.sh"
    exit 1
fi

# Check if Moses path exists
if [ ! -d "$MOSES_PATH" ]; then
    echo "Error: Moses directory not found at $MOSES_PATH"
    echo "Please make sure the environment was set up correctly"
    exit 1
fi

# Process each split
for split in "${SPLITS[@]}"; do
    process_split $split
done

echo "Preprocessing completed for all datasets. Preprocessed files are in the '$OUTPUT_DIR' directory."