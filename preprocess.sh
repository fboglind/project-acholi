#!/bin/bash

# Base path for Moses tools
MOSES_PATH="/common/student/courses/MT-5LN711/tools/MOSES/ubuntu-16.04"

# Input files
ACH_INPUT="salt.ach"
ENG_INPUT="salt.en"

# Tokenization
echo "Tokenizing files..."
$MOSES_PATH/scripts/tokenizer/tokenizer.perl -l en < $ENG_INPUT > salt.tk.eng
$MOSES_PATH/scripts/tokenizer/tokenizer.perl -l en < $ACH_INPUT > salt.tk.ach  # Using English tokenization for Acholi

# Lowercasing
echo "Lowercasing files..."
$MOSES_PATH/scripts/tokenizer/lowercase.perl < salt.tk.eng > salt.train.tk.lc.eng
$MOSES_PATH/scripts/tokenizer/lowercase.perl < salt.tk.ach > salt.train.tk.lc.ach

# Cleaning
echo "Cleaning corpus..."
$MOSES_PATH/scripts/training/clean-corpus-n.perl salt.train.tk.lc ach eng salt.train.tk.lc.fl 1 40  # Corrected input and output filenames

echo "Preprocessing completed."