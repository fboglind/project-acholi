"""
This script does the following:
    Creates a .yaml config-file for pretraining operations
    Encodes data using BPE using subword-nmt
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
    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        vocab_size: int = 8000, # the model will consider the most frequent 8000 subword units from the BPE-encoded data. Reduce if overfitting..
        min_frequency: int = 1, # tokens appearing at least once are included. Reduces out-of-vocabulary issues, resulting in <unk>.
        bpe_operations: int = 8000 # a large number of BPE operations may overfit to training data.
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.bpe_operations = bpe_operations
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
            # Data settings
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
            # Vocabulary settings
            'src_vocab': f"{self.save_data}.vocab.{self.src_lang}",
            'tgt_vocab': f"{self.save_data}.vocab.{self.tgt_lang}",
            'src_vocab_size': self.vocab_size,
            'tgt_vocab_size': self.vocab_size,
            'src_words_min_frequency': self.min_frequency,
            'tgt_words_min_frequency': self.min_frequency,
            'share_vocab': False,
            # Transforms
            'transforms': ['filtertoolong'],
            # Prevent overwriting existing files
            'overwrite': False,
        }

        # Save config
        config_path = os.path.join(self.output_dir, f"{self.save_prefix}_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        self.logger.info(f"Configuration file saved to {config_path}")
        return config_path

    def learn_bpe(self):
        """Learn BPE codes from the combined training data."""
        self.logger.info("Learning BPE codes...")
        bpe_codes_path = os.path.join(self.output_dir, f"{self.save_prefix}.codes")

        # Combine training data
        combined_data = []
        with open(self.files[f"train_{self.src_lang}"], 'r', encoding='utf-8') as src_file:
            combined_data.extend(src_file.readlines())
        with open(self.files[f"train_{self.tgt_lang}"], 'r', encoding='utf-8') as tgt_file:
            combined_data.extend(tgt_file.readlines())

        # Learn BPE codes
        with open(bpe_codes_path, 'w', encoding='utf-8') as codes_file:
            learn_bpe(
                combined_data,
                codes_file,
                num_symbols=self.bpe_operations,
                verbose=False
            )

        self.logger.info(f"BPE codes saved to {bpe_codes_path}")
        self.bpe_codes_path = bpe_codes_path

    def apply_bpe(self):
        """Apply BPE codes to all datasets."""
        self.logger.info("Applying BPE codes...")
        bpe = BPE(open(self.bpe_codes_path, 'r', encoding='utf-8'))

        datasets = [
            ('train', self.files[f"train_{self.src_lang}"], self.src_lang),
            ('train', self.files[f"train_{self.tgt_lang}"], self.tgt_lang),
            ('dev', self.files[f"dev_{self.src_lang}"], self.src_lang),
            ('dev', self.files[f"dev_{self.tgt_lang}"], self.tgt_lang),
        ]

        for split, input_path, lang in datasets:
            output_path = os.path.join(self.output_dir, f"{split}.bpe.{lang}")
            with open(input_path, 'r', encoding='utf-8') as infile, \
                open(output_path, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    outfile.write(bpe.process_line(line))
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
    parser.add_argument('--vocab-size', type=int, default=8000,
                        help='Vocabulary size')
    parser.add_argument('--min-frequency', type=int, default=2,
                        help='Minimum token frequency')
    parser.add_argument('--bpe-operations', type=int, default=32000,
                        help='Number of BPE merge operations')

    args = parser.parse_args()

    # Initialize preprocessor
    preprocessor = ONMTPreprocessor(
        args.src_lang,
        args.tgt_lang,
        args.vocab_size,
        args.min_frequency,
        args.bpe_operations
    )

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

# python preprocess_onmt.py \
#   --train-src processed_data_moses/salt.train.tk.lc.clean.ach \
#   --train-tgt processed_data_moses/salt.train.tk.lc.clean.eng \
#   --dev-src processed_data_moses/salt.dev.tk.lc.ach \
#   --dev-tgt processed_data_moses/salt.dev.tk.lc.eng \
#   --src-lang ach \
#   --tgt-lang en \
#   --output-dir onmt_data \
#   --save-prefix data \
#   --vocab-size 8000 \
#   --min-frequency 1 \
#   --bpe-operations 8000