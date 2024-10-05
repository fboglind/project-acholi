"""extract_data.py contains a script for extracting parallell data from a
multilingual huggingface dataset"""

#If needed run: pip install datasets
import os
from datasets import load_dataset

def prepare_data():
    # Load multilingual dataset
    text_all = load_dataset("Sunbird/salt", "text-all", split="train")

    # Prepare output files
    ach_file = os.path.join("salt.ach")
    eng_file = os.path.join("salt.en")

    # Write aligned sentences to files
    with open(ach_file, 'w', encoding='utf-8') as f_ach, open(eng_file, 'w', encoding='utf-8') as f_eng:
        for line in text_all:
            f_ach.write(line['ach_text'] + '\n')
            f_eng.write(line['eng_text'] + '\n')

    print(f"Data prepared and saved:")
    print(f"Acholi sentences: {ach_file}")
    print(f"English sentences: {eng_file}")

if __name__ == '__main__':
    prepare_data()
