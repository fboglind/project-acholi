"""Preprocessing using NLTK"""
import os
import nltk
from nltk.tokenize import word_tokenize
from pathlib import Path
import argparse
from typing import List, Tuple
import re

# Download required NLTK data
nltk.download('punkt_tab')

class Preprocessor:
    def __init__(self, input_dir: str, output_dir: str):
        """Initialize preprocessor with input and output directories."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.splits = ["train", "dev", "test"]

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def tokenize_text(self, text: str, lang: str) -> List[str]:
        """Tokenize text using NLTK's word_tokenize."""
        # You might want to customize tokenization rules based on language
        return word_tokenize(text)

    def clean_corpus(self, source_lines: List[List[str]], target_lines: List[List[str]], 
                    min_len: int = 1, max_len: int = 40) -> Tuple[List[List[str]], List[List[str]]]:
        """Clean parallel corpus by filtering based on length constraints."""
        cleaned_source = []
        cleaned_target = []
        
        for src_tokens, tgt_tokens in zip(source_lines, target_lines):
            if min_len <= len(src_tokens) <= max_len and min_len <= len(tgt_tokens) <= max_len:
                cleaned_source.append(src_tokens)
                cleaned_target.append(tgt_tokens)
                
        return cleaned_source, cleaned_target

    def process_file(self, input_path: Path, lang: str) -> List[List[str]]:
        """Process a single file: read, tokenize, and lowercase.
        Returns list of tokenized sentences (list of lists)."""
        tokenized_lines = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Tokenize each line
                tokens = self.tokenize_text(line.strip(), lang)
                # Lowercase
                tokens = [token.lower() for token in tokens]
                tokenized_lines.append(tokens)
                
        return tokenized_lines

    def save_tokens(self, tokenized_lines: List[List[str]], output_path: Path):
        """Save tokenized text to file, maintaining line breaks."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for tokens in tokenized_lines:
                f.write(' '.join(tokens) + '\n')

    def process_split(self, split: str):
        """Process a single data split."""
        print(f"Processing {split} set...")

        # Input files
        ach_input = self.input_dir / f"salt.{split}.ach"
        eng_input = self.input_dir / f"salt.{split}.en"

        # Process both languages
        ach_tokens = self.process_file(ach_input, 'ach')
        eng_tokens = self.process_file(eng_input, 'en')

        # Clean corpus for training data
        if split == "train":
            ach_tokens, eng_tokens = self.clean_corpus(ach_tokens, eng_tokens)

        # Save tokenized and lowercased versions
        self.save_tokens(ach_tokens,
                        self.output_dir / f"salt.{split}.tk.lc.ach")
        self.save_tokens(eng_tokens,
                        self.output_dir / f"salt.{split}.tk.lc.eng")

        print(f"Completed processing {split} set")

    def process_all(self):
        """Process all data splits."""
        for split in self.splits:
            self.process_split(split)
        print(f"Preprocessing completed. Processed files are in '{self.output_dir}'")

def main():
    parser = argparse.ArgumentParser(description='Preprocess data using NLTK')
    parser.add_argument('--input-dir', default='data',
                       help='Input directory containing source files')
    parser.add_argument('--output-dir', default='processed_data_nltk',
                       help='Output directory for processed files')

    args = parser.parse_args()

    preprocessor = Preprocessor(args.input_dir, args.output_dir)
    preprocessor.process_all()

if __name__ == "__main__":
    main()
