# overlap_analysis.py
import os
from datasets import load_dataset
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.metrics import edit_distance

import pybktree


#pip install pybktree
#pip install datasets nltk


# If necessary, run pip install NLTK
# Download NLTK data files (only needed the first time)
nltk.download('punkt')
nltk.download('punkt_tab')

def load_and_tokenize_data(language_code):
    # Load the 'text-all' configuration
    data = load_dataset("Sunbird/salt", "text-all", split="train+dev+test")
    
    # The text field names are '{language_code}_text', e.g., 'ach_text', 'nyn_text'
    text_field = f"{language_code}_text"
    
    # Extract the text data for the language, ensuring the field exists
    text_data = [item[text_field] for item in data if item.get(text_field)]
    
    # Tokenize and normalize the text data
    tokens = []
    for sentence in text_data:
        # Tokenize the sentence
        words = word_tokenize(sentence.lower())
        # Remove punctuation and non-alphabetic tokens
        words = [word for word in words if word.isalpha()]
        tokens.extend(words)
    return tokens

def build_vocabulary(tokens, top_n=None):
    # Build a frequency count of tokens
    token_counts = Counter(tokens)
    if top_n:
        # Keep only the top N most common words
        most_common = token_counts.most_common(top_n)
        vocab = set(word for word, count in most_common)
    else:
        vocab = set(token_counts.keys())
    return vocab


def compute_word_overlap(vocab_acholi, vocab_other):
    # Compute the intersection of vocabularies
    common_words = vocab_acholi.intersection(vocab_other)
    # Compute Jaccard similarity
    jaccard_similarity = len(common_words) / len(vocab_acholi.union(vocab_other))
    return common_words, jaccard_similarity

def find_similar_words(vocab_acholi, vocab_other, max_distance=1):
    # Build a BK-tree with the other language's vocabulary
    print("Building BK-tree for the other language's vocabulary...")
    tree = pybktree.BKTree(edit_distance, vocab_other)
    similar_words = []
    print("Searching for similar words...")
    for word in vocab_acholi:
        matches = tree.find(word, max_distance)
        for distance, match in matches:
            similar_words.append((word, match, distance))
    return similar_words
def main():
    # Language codes
    languages = ["nyn", "swa", "teo", "ibo", "lgg", "lug", "eng"]
    acholi_code = "ach"
    
    # Load and tokenize Acholi data
    print("Loading and tokenizing Acholi data...")
    tokens_acholi = load_and_tokenize_data(acholi_code)
    vocab_acholi = build_vocabulary(tokens_acholi, top_n=5000)
    print(f"Acholi vocabulary size: {len(vocab_acholi)} words")
    
    # Initialize a dictionary to store overlap results
    overlap_results = {}
    
    for lang_code in languages:
        print(f"\nProcessing language: {lang_code}")
        # Load and tokenize other language data
        tokens_other = load_and_tokenize_data(lang_code)
        vocab_other = build_vocabulary(tokens_other)
        print(f"{lang_code} vocabulary size: {len(vocab_other)} words")
        
        # Compute word overlap
        common_words, jaccard_similarity = compute_word_overlap(vocab_acholi, vocab_other)
        print(f"Number of overlapping words with Acholi: {len(common_words)}")
        print(f"Jaccard similarity with Acholi: {jaccard_similarity:.4f}")
        
        # Store results
        overlap_results[lang_code] = {
            "common_words": common_words,
            "jaccard_similarity": jaccard_similarity
        }
        
        # Optionally compute similar words
        print("Computing similar words based on edit distance...")
        similar_words = find_similar_words(vocab_acholi, vocab_other, max_distance=1)
        print(f"Number of similar words (edit distance <= 1): {len(similar_words)}")
        
        # Save or analyze similar_words as needed
        # For example, save to a file
        output_file = f"similar_words_ach_{lang_code}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for word_pair in similar_words:
                f.write(f"{word_pair[0]}\t{word_pair[1]}\tDistance: {word_pair[2]}\n")
        print(f"Similar words saved to {output_file}")
    
    # Optional: Save overlap results to a file
    with open("overlap_results.txt", 'w', encoding='utf-8') as f:
        for lang_code, result in overlap_results.items():
            f.write(f"Language: {lang_code}\n")
            f.write(f"Jaccard Similarity: {result['jaccard_similarity']:.4f}\n")
            f.write(f"Number of overlapping words: {len(result['common_words'])}\n")
            f.write("\n")
    print("\nOverlap results saved to overlap_results.txt")

if __name__ == "__main__":
    main()
