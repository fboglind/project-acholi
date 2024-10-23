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
    output_dir = 'data_all'
    os.makedirs(output_dir, exist_ok=True)

    # Function to write data to files
    def write_to_files(data, ach_file, eng_file):
        with open(ach_file, 'w', encoding='utf-8') as f_ach, open(eng_file, 'w', encoding='utf-8') as f_eng:
            for item in data:
                f_ach.write(item['ach_text'] + '\n')
                f_eng.write(item['eng_text'] + '\n')

    # Write train data to files
    ach_path = os.path.join(output_dir, "salt.train.ach")
    eng_path = os.path.join(output_dir, "salt.train.en")
    
    try:
        write_to_files(train_data, ach_path, eng_path)
        print(f"Data prepared and saved in {output_dir}:")
        print(f"Training set: salt.train.ach, salt.train.en ({len(train_data)} pairs)")
    except Exception as e:
        print(f"Error occurred while writing files: {e}")

if __name__ == '__main__':
    prepare_data()