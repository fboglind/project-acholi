"""load_all_data.py contains a script for extracting parallel data from a
multilingual dataset (https://huggingface.co/datasets/Sunbird/salt).
The data is extraxted as one single training set"""
# If needed run: pip install datasets
import os
from datasets import load_dataset

def prepare_data():
    # Load multilingual dataset for each split
    train_data = load_dataset("Sunbird/salt", "text-all", split="train")

# Create data directory if it does not exist
    os.makedirs('data_all', exist_ok=True)

    # Function to write data to files
    def write_to_files(data, ach_file, eng_file):
        with open(ach_file, 'w', encoding='utf-8') as f_ach, open(eng_file, 'w', encoding='utf-8') as f_eng:
            for item in data:
                f_ach.write(item['ach_text'] + '\n')
                f_eng.write(item['eng_text'] + '\n')

    # Write train, dev, and test data to files
    write_to_files(train_data, "data_all/salt.train.ach", "data/salt.train.en")
   

    print("Data prepared and saved in /data_all:")
    print(f"Training set: salt.train.ach, salt.train.en ({len(train_data)} pairs)")

if __name__ == '__main__':
    prepare_data()