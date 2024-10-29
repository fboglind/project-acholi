"""
This script does the following:
    Creates a .yaml config-file for pretraining operations
    Encodes data using Byte Pair Encoding (subword-nmt)
    Creates a vocabulary for use with OpenNMT
"""
import argparse
import yaml
import logging
import subprocess
import os
from typing import Dict
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE


class ONMTPreprocessor:
    # def __init__(
    #     self,
    #     src_lang: str,
    #     tgt_lang: str,                     #RUN 2:
    #     src_vocab_size: int = 5000,        # Reduced for Acholi
    #     tgt_vocab_size: int = 6000,        # Slightly higher for English
    #     src_min_frequency: int = 3,        # Increased to reduce noise
    #     tgt_min_frequency: int = 2,        # Keep as is for English
    #     src_bpe_operations: int = 5000,    # Match vocab size
    #     tgt_bpe_operations: int = 6000     # Match vocab size
    def __init__(
        self,
        src_lang: str,                #RUN 1
        tgt_lang: str,
        src_vocab_size: int = 7000,
        tgt_vocab_size: int = 7000,    
        src_min_frequency: int = 2,        
        tgt_min_frequency: int = 2, 
        src_bpe_operations: int = 7000,    
        tgt_bpe_operations: int = 7000     # larger number of BPE operations may overfit to training data.

    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_min_frequency = src_min_frequency
        self.tgt_min_frequency = tgt_min_frequency
        self.src_bpe_operations = src_bpe_operations
        self.tgt_bpe_operations = tgt_bpe_operations
        
        self.files: Dict[str, str] = {}

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def set_file_paths(
        self,
        train_src: str,
        train_tgt: str,
        dev_src: str,
        dev_tgt: str,
        output_dir: str,
        save_prefix: str
    ):
        """Store all file paths."""
        self.files = {
            f"train_{self.src_lang}": train_src,
            f"train_{self.tgt_lang}": train_tgt,
            f"dev_{self.src_lang}": dev_src,
            f"dev_{self.tgt_lang}": dev_tgt
        }
        self.output_dir = output_dir
        self.save_prefix = save_prefix

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Set the save_data path
        self.save_data = os.path.join(self.output_dir, self.save_prefix)

    def create_yaml_config(self) -> str:
        """Create YAML configuration for preprocessing."""
        config = {
            'save_data': self.save_data,
            'data': {
                'corpus_1': {
                    'path_src': os.path.join(self.output_dir, f"train.bpe.{self.src_lang}"),
                    'path_tgt': os.path.join(self.output_dir, f"train.bpe.{self.tgt_lang}"),
                },
                'valid': {
                    'path_src': os.path.join(self.output_dir, f"dev.bpe.{self.src_lang}"),
                    'path_tgt': os.path.join(self.output_dir, f"dev.bpe.{self.tgt_lang}"),
                },
            },
            'src_vocab': f"{self.save_data}.vocab.{self.src_lang}",
            'tgt_vocab': f"{self.save_data}.vocab.{self.tgt_lang}",
            'src_vocab_size': self.src_vocab_size,      # using language-specific size
            'tgt_vocab_size': self.tgt_vocab_size, 
            'src_words_min_frequency': self.src_min_frequency,  # language-specific minimum frequency
            'tgt_words_min_frequency': self.tgt_min_frequency,
            'share_vocab': False, # Shared might also work
            'transforms': ['filtertoolong'],
            'overwrite': False, # prevent overwriting existing files
        }

        # Save config
        config_path = os.path.join(self.output_dir, f"{self.save_prefix}_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        self.logger.info(f"Configuration file saved to {config_path}")
        return config_path

    def learn_bpe(self):
        """Learn separate BPE codes for source and target languages."""
        self.logger.info("Learning BPE codes...")
        
        # Create separate BPE codes files for each language
        src_bpe_codes_path = os.path.join(self.output_dir, f"{self.save_prefix}.{self.src_lang}.codes")
        tgt_bpe_codes_path = os.path.join(self.output_dir, f"{self.save_prefix}.{self.tgt_lang}.codes")
        
        # Learn BPE for source language
        self.logger.info(f"Learning BPE for {self.src_lang}...")
        with open(self.files[f"train_{self.src_lang}"], 'r', encoding='utf-8') as src_file:
            src_data = src_file.readlines()
            with open(src_bpe_codes_path, 'w', encoding='utf-8') as codes_file:
                learn_bpe(
                src_data,
                codes_file,
                num_symbols=self.src_bpe_operations,  # use source-specific BPE operations
                verbose=False
            )
        
        # Learn BPE for target language
        self.logger.info(f"Learning BPE for {self.tgt_lang}...")
        with open(self.files[f"train_{self.tgt_lang}"], 'r', encoding='utf-8') as tgt_file:
            tgt_data = tgt_file.readlines()
            with open(tgt_bpe_codes_path, 'w', encoding='utf-8') as codes_file:
                learn_bpe(
                tgt_data,
                codes_file,
                num_symbols=self.tgt_bpe_operations,
                verbose=False
            )
                    
        # Store paths for use in apply_bpe
        self.src_bpe_codes_path = src_bpe_codes_path
        self.tgt_bpe_codes_path = tgt_bpe_codes_path

    def apply_bpe(self):
        """Apply language-specific BPE codes to datasets."""
        self.logger.info("Applying BPE codes...")
        
        # Create BPE processors for each language
        src_bpe = BPE(open(self.src_bpe_codes_path, 'r', encoding='utf-8'))
        tgt_bpe = BPE(open(self.tgt_bpe_codes_path, 'r', encoding='utf-8'))

        datasets = [
            ('train', self.files[f"train_{self.src_lang}"], self.src_lang, src_bpe),
            ('train', self.files[f"train_{self.tgt_lang}"], self.tgt_lang, tgt_bpe),
            ('dev', self.files[f"dev_{self.src_lang}"], self.src_lang, src_bpe),
            ('dev', self.files[f"dev_{self.tgt_lang}"], self.tgt_lang, tgt_bpe),
        ]

        for split, input_path, lang, bpe_processor in datasets:
            output_path = os.path.join(self.output_dir, f"{split}.bpe.{lang}")
            with open(input_path, 'r', encoding='utf-8') as infile, \
                open(output_path, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    outfile.write(bpe_processor.process_line(line))
            self.logger.info(f"BPE applied to {input_path}, output saved to {output_path}")

    def build_vocab(self):
        """Build vocabulary using onmt_build_vocab."""
        try:
            self.logger.info("Building vocabulary...")
            config_path = os.path.join(self.output_dir, f"{self.save_prefix}_config.yaml")
            build_vocab_cmd = [
                "onmt_build_vocab",
                "-config", config_path,
                "-n_sample", str(10000)  # Adjust as needed
            ]
            subprocess.run(build_vocab_cmd, check=True)
            self.logger.info("Vocabulary building completed successfully.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error during vocabulary building: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess data using OpenNMT-py tools with BPE encoding'
    )

    parser.add_argument('--train-src', required=True,
                        help='Training source file')
    parser.add_argument('--train-tgt', required=True,
                        help='Training target file')
    parser.add_argument('--dev-src', required=True,
                        help='Validation source file')
    parser.add_argument('--dev-tgt', required=True,
                        help='Validation target file')
    parser.add_argument('--output-dir', default='onmt_data',
                        help='Directory where output files will be saved')
    parser.add_argument('--save-prefix', default='data',
                        help='Prefix for saved data files')
    parser.add_argument('--src-lang', default='src',
                        help='Source language code')
    parser.add_argument('--tgt-lang', default='tgt',
                        help='Target language code')
    parser.add_argument('--src-vocab-size', type=int, default=4000,
                        help='Source vocabulary size')
    parser.add_argument('--tgt-vocab-size', type=int, default=8000,
                        help='Target vocabulary size')
    parser.add_argument('--src-min-frequency', type=int, default=2,
                        help='Minimum token frequency for source')
    parser.add_argument('--tgt-min-frequency', type=int, default=1,
                        help='Minimum token frequency for target')
    parser.add_argument('--src-bpe-operations', type=int, default=4000,
                        help='Number of BPE merge operations for source')
    parser.add_argument('--tgt-bpe-operations', type=int, default=8000,
                        help='Number of BPE merge operations for target')

    args = parser.parse_args()

    # Initialize preprocessor with new parameters
    preprocessor = ONMTPreprocessor(
        args.src_lang,
        args.tgt_lang,
        args.src_vocab_size,
        args.tgt_vocab_size,
        args.src_min_frequency,
        args.tgt_min_frequency,
        args.src_bpe_operations,
        args.tgt_bpe_operations
    )

    args = parser.parse_args()

    # # Initialize preprocessor
    # preprocessor = ONMTPreprocessor(
    #     args.src_lang,
    #     args.tgt_lang,
    #     args.vocab_size,
    #     args.min_frequency,
    #     args.bpe_operations
    # )

    # Set file paths with the output directory and prefix
    preprocessor.set_file_paths(
        os.path.abspath(args.train_src),
        os.path.abspath(args.train_tgt),
        os.path.abspath(args.dev_src),
        os.path.abspath(args.dev_tgt),
        args.output_dir,
        args.save_prefix
    )

    # Learn BPE codes
    preprocessor.learn_bpe()

    # Apply BPE codes
    preprocessor.apply_bpe()

    # Create YAML configuration
    preprocessor.create_yaml_config()

    # Build Vocabulary
    preprocessor.build_vocab()


if __name__ == "__main__":
    main()

# Command used for baseline:

## Run 1
# python preprocess_onmt.py \
#   --train-src processed_data_moses/salt.train.tk.lc.clean.ach \
#   --train-tgt processed_data_moses/salt.train.tk.lc.clean.eng \
#   --dev-src processed_data_moses/salt.dev.tk.lc.ach \
#   --dev-tgt processed_data_moses/salt.dev.tk.lc.eng \
#   --src-lang ach \
#   --tgt-lang en \
#   --output-dir onmt_data \
#   --save-prefix data \
#   --src-vocab-size 7000 \
#   --tgt-vocab-size 7000 \
#   --src-min-frequency 1 \
#   --tgt-min-frequency 1 \
#   --src-bpe-operations 7000 \
#   --tgt-bpe-operations 7000

## Run 2
# python preprocess_onmt.py \
#   --train-src processed_data_moses/salt.train.tk.lc.clean.ach \
#   --train-tgt processed_data_moses/salt.train.tk.lc.clean.eng \
#   --dev-src processed_data_moses/salt.dev.tk.lc.ach \
#   --dev-tgt processed_data_moses/salt.dev.tk.lc.eng \
#   --src-lang ach \
#   --tgt-lang en \
#   --output-dir onmt_data \
#   --save-prefix data \
#   --src-vocab-size 5000 \
#   --tgt-vocab-size 6000 \
#   --src-min-frequency 3 \
#   --tgt-min-frequency 2 \
#   --src-bpe-operations 5000 \
#   --tgt-bpe-operations 6000