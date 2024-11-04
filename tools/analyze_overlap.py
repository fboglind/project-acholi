"""analyze_overlap.py contains tools for analyzing overlap in parallel language data files"""
import os
import argparse
from collections import Counter

MIN_WORD_LENGTH = 4  # Minimum word length to consider

def load_corpus(file_path, min_length):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [[word.lower() for word in line.strip().split() if len(word) >= min_length] for line in f]

def get_word_overlap(ach_corpus, eng_corpus):
    ach_words = set(word for sentence in ach_corpus for word in sentence)
    eng_words = set(word for sentence in eng_corpus for word in sentence)
    return ach_words.intersection(eng_words)

def analyze_overlap(ach_corpus, eng_corpus, overlap, output_file):
    overlap_count = Counter()
    total_ach_words = sum(len(sentence) for sentence in ach_corpus)
    overlap_instances = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (ach_sentence, eng_sentence) in enumerate(zip(ach_corpus, eng_corpus)):
            sentence_overlap = [word for word in ach_sentence if word in overlap]
            if sentence_overlap:
                f.write(f"Line {i+1}:\n")
                f.write(f"Acholi:  {' '.join(ach_sentence)}\n")
                f.write(f"English: {' '.join(eng_sentence)}\n")
                f.write(f"Overlapping words: {', '.join(sentence_overlap)}\n\n")
                overlap_instances += len(sentence_overlap)
            for word in sentence_overlap:
                overlap_count[word] += 1

    return overlap_count, total_ach_words, overlap_instances

def write_stats(stats, output_file, data_type):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Analysis for {data_type} data (words with length >= {MIN_WORD_LENGTH})\n\n")
        for split, data in stats.items():
            f.write(f"\nStatistics for {split} set:\n")
            f.write(f"Total Acholi words (length >= {MIN_WORD_LENGTH}): {data['total_words']}\n")
            f.write(f"Unique Acholi words: {data['unique_ach_words']}\n")
            f.write(f"Unique English words: {data['unique_eng_words']}\n")
            f.write(f"Number of overlapping word types: {data['overlap_types']}\n")
            f.write(f"Number of overlapping word instances: {data['overlap_instances']}\n")
            f.write(f"Percentage of overlapping word types: {data['overlap_types_percent']:.2f}%\n")
            f.write(f"Percentage of overlapping word instances: {data['overlap_instances_percent']:.2f}%\n")
            f.write("\nTop 10 overlapping words:\n")
            for word, count in data['top_overlap']:
                f.write(f"{word}: {count}\n")
            f.write("\n")

def main(data_type):
    if data_type == 'raw':
        data_dir = "data"
        file_prefix = "salt"
    else:
        data_dir = "processed"
        file_prefix = "salt.train.tk.lc"

    output_dir = f"overlap_analysis_{data_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    splits = ["train", "dev", "test"] if data_type == 'raw' else ["train"]
    stats = {}

    for split in splits:
        print(f"Analyzing {split} set...")
        if data_type == 'raw':
            ach_path = os.path.join(data_dir, f"{file_prefix}.{split}.ach")
            eng_path = os.path.join(data_dir, f"{file_prefix}.{split}.en")
        else:
            ach_path = os.path.join(data_dir, f"{file_prefix}.ach")
            eng_path = os.path.join(data_dir, f"{file_prefix}.eng")

        ach_corpus = load_corpus(ach_path, MIN_WORD_LENGTH)
        eng_corpus = load_corpus(eng_path, MIN_WORD_LENGTH)

        overlap = get_word_overlap(ach_corpus, eng_corpus)
        output_file = os.path.join(output_dir, f"{split}_overlap_examples.txt")
        overlap_count, total_words, overlap_instances = analyze_overlap(ach_corpus, eng_corpus, overlap, output_file)

        unique_ach_words = len(set(word for sentence in ach_corpus for word in sentence))
        unique_eng_words = len(set(word for sentence in eng_corpus for word in sentence))

        stats[split] = {
            'total_words': total_words,
            'unique_ach_words': unique_ach_words,
            'unique_eng_words': unique_eng_words,
            'overlap_types': len(overlap),
            'overlap_instances': overlap_instances,
            'overlap_types_percent': (len(overlap) / unique_ach_words) * 100 if unique_ach_words > 0 else 0,
            'overlap_instances_percent': (overlap_instances / total_words) * 100 if total_words > 0 else 0,
            'top_overlap': overlap_count.most_common(10)
        }

    write_stats(stats, os.path.join(output_dir, "overlap_statistics.txt"), data_type)
    print(f"Analysis complete. Results saved in the '{output_dir}' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze word overlap in Acholi-English bitext.")
    parser.add_argument('--data_type', choices=['raw', 'processed'], default='raw',
                        help="Type of data to analyze: 'raw' or 'processed'")
    args = parser.parse_args()
    main(args.dat
