# project-acholi-mt24
This repository hosts code for a low-resource language machine translation project involving the Acholi language. The project is a part of the course Machine Translation (5LN711) at Uppsala University, Fall 2024.

To clone the depository use:
```
git clone https://github.com/project-acholi-mt24/project-acholi-mt24.git
```
See shared folder on UPPMAX for data files used for baseline training:  /proj/uppmax2024-2-13/private/acholi_mt24/project-acholi-mt24

# Training the baseline model
### Step 1: Create virtual environment

1. In the appropriate directory (On UPPMAX ),
run the script for creating a virtual environment:
```
bash create_mt_env.sh
```

2. Activate the environment: 
```
source ~/envs/activate_acholi_mt_env.sh
```
3. When done, deactivate environment:
```
deactivate
```
### Step 2: Extract data for baseline model

Run extract_data.py:
```
python extract_data.py
```
### Step 3: Tokenize and clean the data
Run preprocess.sh:
```
bash preprocess.sh
```
### Step 4: Create vocabulary, Encode data using BPE, create .yaml-file with data configuration

Run preprocess_onmt.py:
Example:
```
python preprocess_onmt.py \
  --train-src processed_data_moses/salt.train.tk.lc.clean.ach \
  --train-tgt processed_data_moses/salt.train.tk.lc.clean.eng \
  --dev-src processed_data_moses/salt.dev.tk.lc.ach \
  --dev-tgt processed_data_moses/salt.dev.tk.lc.eng \
  --src-lang ach \
  --tgt-lang en \
  --output-dir onmt_data \
  --save-prefix data \
  --src-vocab-size 7000 \
  --tgt-vocab-size 7000 \
  --src-min-frequency 2 \
  --tgt-min-frequency 2 \
  --src-bpe-operations 7000 \
  --tgt-bpe-operations 7000
```
### Step 5: *Manually* create the file train_config.yaml
Use the newly created file *data_config.yaml* as a base. Set parameters. See *train_config.yaml.example*...

### Step 6: Submit job to Snowy (Uppmax cluster):
```
sbatch train_baseline_model.sh
```

### Step 7: Translate using the newly trained model [WILL BE UPDATED]
Use the the newly trained model to translate the data in the test set
```
sbatch translate_model.sh
```
These parameters can be useful (for the *onmt_translate* command IN the translate_model.sh-file):
- batch_size: Number of sentences to translate at once
- beam_size: Size of beam search (larger = potentially better but slower)
- replace_unk: Replaces unknown tokens with source tokens
- Add -n_best 3 for multiple translations per sentence
- Add -fp32 if you experience any precision issues
________________________________________________________________________________________________________

# Scripts for data extraction and tokenization/initial preprocessing

### extract_data.py

This script extracts parallel data from the Sunbird/salt dataset on Hugging Face.

**Usage:**
```
python extract_data.py
```

This script will extract raw Acholi-English parallel data

### preprocess.sh

This bash script preprocesses the extracted data for use with Moses SMT.

**Usage:**
```
./preprocess.sh
```

This script will:
- Tokenize the Acholi and English texts
- Lowercase all tokens
- Clean the training corpus (remove long sentences and empty lines)
- Save the preprocessed files
## Setup

1. Clone this repository:
   ```
   git clone https://github.com/project-acholi-mt24/project-acholi-mt24.git
   cd project-acholi-mt24
   ```

2. Install required Python packages:
   ```
   pip install datasets
   ```

3. Run the data extraction script:
   ```
   python extract_data.py
   ```

4. Run the preprocessing script:
   ```
   chmod +x preprocess.sh
   ./preprocess.sh
   ```
______________________________________________________________________________________________

Some more files:

### create_mt_env.sh

This script will:
- Create virtual environment on the server (UPPMAX) (see details above)

### preprocess_onmt.py:

This script will:

- Create a .yaml config-file for pretraining operations
- Encode data using BPE using subword-nmt
- Create a vocabulary for use with OpenNMT

### analyze_line_endings.py:

This script will:
- Compare line endings of two text files and analyze punctuation patterns.
- Return statistics about matching and mismatching line endings.

### train_baseline_model.sh

This script will:
- Load a batch job on the server in order to train a baseline model

### preprocess.py

This script will:
- Preprocess/Tokenize using nltk
