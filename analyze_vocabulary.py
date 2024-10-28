import argparse
from collections import Counter
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

# pip install matplotlib

def analyze_vocab(filepath: str) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Analyze vocabulary and token frequencies in a file.
    
    Returns:
        Tuple containing:
        - Dictionary with basic stats
        - Dictionary with frequency distribution
    """
    # Read file and collect statistics
    tokens = []
    token_count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line_tokens = line.strip().split()
            tokens.extend(line_tokens)
            token_count += len(line_tokens)
    
    # Count unique tokens
    token_freqs = Counter(tokens)
    
    # Calculate statistics
    stats = {
        'total_tokens': token_count,
        'unique_tokens': len(token_freqs),
        'tokens_occurring_once': sum(1 for t, c in token_freqs.items() if c == 1),
        'tokens_occurring_twice': sum(1 for t, c in token_freqs.items() if c == 2),
        'tokens_occurring_5+_times': sum(1 for t, c in token_freqs.items() if c >= 5),
    }
    
    # Calculate frequency distribution for plotting
    freq_dist = {}
    for threshold in [1, 2, 3, 5, 10, 20, 50, 100]:
        freq_dist[threshold] = sum(1 for t, c in token_freqs.items() if c >= threshold)
    
    return stats, freq_dist

def main():
    parser = argparse.ArgumentParser(description='Analyze vocabulary statistics for two languages')
    parser.add_argument('--src', required=True, help='Source language file')
    parser.add_argument('--tgt', required=True, help='Target language file')
    parser.add_argument('--src-name', default='Source', help='Name of source language')
    parser.add_argument('--tgt-name', default='Target', help='Name of target language')
    args = parser.parse_args()
    
    # Analyze both files
    src_stats, src_freq = analyze_vocab(args.src)
    tgt_stats, tgt_freq = analyze_vocab(args.tgt)
    
    # Print basic statistics
    print(f"\nVocabulary Statistics:")
    print(f"{'Metric':<25} {args.src_name:<15} {args.tgt_name:<15}")
    print("-" * 55)
    
    metrics = [
        ('Total tokens', 'total_tokens'),
        ('Unique tokens', 'unique_tokens'),
        ('Tokens occurring once', 'tokens_occurring_once'),
        ('Tokens occurring twice', 'tokens_occurring_twice'),
        ('Tokens occurring 5+ times', 'tokens_occurring_5+_times')
    ]
    
    for metric_name, metric_key in metrics:
        print(f"{metric_name:<25} {src_stats[metric_key]:<15,d} {tgt_stats[metric_key]:<15,d}")
    
    # Print vocabulary ratios
    print(f"\nVocabulary Analysis:")
    print(f"- {args.src_name} type-token ratio: {src_stats['unique_tokens']/src_stats['total_tokens']:.4f}")
    print(f"- {args.tgt_name} type-token ratio: {tgt_stats['unique_tokens']/tgt_stats['total_tokens']:.4f}")
    print(f"- Ratio of {args.tgt_name} to {args.src_name} unique tokens: {tgt_stats['unique_tokens']/src_stats['unique_tokens']:.2f}")
    
    # Plot frequency distribution
    thresholds = list(src_freq.keys())
    src_values = list(src_freq.values())
    tgt_values = list(tgt_freq.values())
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(thresholds))
    width = 0.35
    
    plt.bar(x - width/2, src_values, width, label=args.src_name)
    plt.bar(x + width/2, tgt_values, width, label=args.tgt_name)
    
    plt.xlabel('Minimum Frequency Threshold')
    plt.ylabel('Number of Tokens')
    plt.title('Vocabulary Size at Different Frequency Thresholds')
    plt.xticks(x, thresholds)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig('vocab_analysis.png')
    print("\nPlot saved as 'vocab_analysis.png'")

if __name__ == "__main__":
    main()

# python analyze_vocabulary.py \
#   --src processed_data_moses/salt.train.tk.lc.clean.ach \
#   --tgt processed_data_moses/salt.train.tk.lc.clean.eng \
#   --src-name Acholi \
#   --tgt-name English