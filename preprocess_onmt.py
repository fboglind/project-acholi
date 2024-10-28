import argparse
import yaml
import logging
import subprocess
import os
from typing import Dict

class ONMTPreprocessor:
    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        vocab_size: int = 8000,
        min_frequency: int = 2
    ):
        """Initialize OpenNMT preprocessor."""
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
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
                    'path_src': self.files[f"train_{self.src_lang}"],
                    'path_tgt': self.files[f"train_{self.tgt_lang}"],
                },
                'valid': {
                    'path_src': self.files[f"dev_{self.src_lang}"],
                    'path_tgt': self.files[f"dev_{self.tgt_lang}"],
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
            # Prevent overwriting existing files
            'overwrite': False,
        }

        # Save config
        config_path = os.path.join(self.output_dir, f"{self.save_prefix}_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        self.logger.info(f"Configuration file saved to {config_path}")
        return config_path

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
        description='Preprocess data using OpenNMT-py tools'
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
    parser.add_argument('--save-prefix', default='vocab',
                        help='Prefix for saved data files')
    parser.add_argument('--src-lang', default='src',
                        help='Source language code')
    parser.add_argument('--tgt-lang', default='tgt',
                        help='Target language code')
    parser.add_argument('--vocab-size', type=int, default=8000,
                        help='Vocabulary size')
    parser.add_argument('--min-frequency', type=int, default=2,
                        help='Minimum token frequency')

    args = parser.parse_args()

    # Set absolute file paths:
    train_src = os.path.abspath(args.train_src)
    train_tgt = os.path.abspath(args.train_tgt)
    dev_src = os.path.abspath(args.dev_src)
    dev_tgt = os.path.abspath(args.dev_tgt)

    # Initialize preprocessor
    preprocessor = ONMTPreprocessor(
        args.src_lang,
        args.tgt_lang,
        args.vocab_size,
        args.min_frequency
    )

    # Set file paths with the output directory and prefix
    preprocessor.set_file_paths(
        args.train_src,
        args.train_tgt,
        args.dev_src,
        args.dev_tgt,
        args.output_dir,
        args.save_prefix
    )

    # Create YAML configuration
    preprocessor.create_yaml_config()

    # Build Vocabulary
    preprocessor.build_vocab()


if __name__ == "__main__":
    main()
