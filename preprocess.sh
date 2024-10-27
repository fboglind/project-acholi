#!/bin/bash

# Base path for Moses tools
MOSES_PATH="/common/student/courses/MT-5LN711/tools/MOSES/ubuntu-16.04"

# Input and output directories
DATA_DIR="data"
OUTPUT_DIR="processed_data_moses"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

SPLITS=("train" "dev" "test")

# Function to process a single split
process_split() {
    local split=$1
    echo "Processing $split set..."

    # Input files
    ACH_INPUT="$DATA_DIR/salt.$split.ach"
    ENG_INPUT="$DATA_DIR/salt.$split.en"

    # Tokenization
    echo "Tokenizing $split files..."
    $MOSES_PATH/scripts/tokenizer/tokenizer.perl -l en < $ENG_INPUT > $OUTPUT_DIR/salt.$split.tk.eng
    $MOSES_PATH/scripts/tokenizer/tokenizer.perl -l en < $ACH_INPUT > $OUTPUT_DIR/salt.$split.tk.ach

    # Lowercasing
    echo "Lowercasing $split files..."
    $MOSES_PATH/scripts/tokenizer/lowercase.perl < $OUTPUT_DIR/salt.$split.tk.eng > $OUTPUT_DIR/salt.$split.tk.lc.eng
    $MOSES_PATH/scripts/tokenizer/lowercase.perl < $OUTPUT_DIR/salt.$split.tk.ach > $OUTPUT_DIR/salt.$split.tk.lc.ach

    # Truecasing
    # echo "Truecasing $split files..."
    # $MOSES_PATH/scripts/tokenizer/truecase.perl < $OUTPUT_DIR/salt.$split.tk.eng > $OUTPUT_DIR/salt.$split.tk.tc.eng
    # $MOSES_PATH/scripts/tokenizer/truecase.perl < $OUTPUT_DIR/salt.$split.tk.ach > $OUTPUT_DIR/salt.$split.tk.tc.ach

    # Cleaning (only for train set)
    if [ "$split" == "train" ]; then
        echo "Cleaning $split corpus..."
        $MOSES_PATH/scripts/training/clean-corpus-n.perl $OUTPUT_DIR/salt.$split.tk.lc ach eng $OUTPUT_DIR/salt.$split.tk.lc.clean 1 40
    fi
}

# Process each split
for split in "${SPLITS[@]}"; do
    process_split $split
done

echo "Preprocessing completed for all datasets. Preprocessed files are in the '$OUTPUT_DIR' directory."
