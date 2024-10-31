"""A tool for comparing end of line punctuation in parallel text files"""

def analyze_line_endings(file1_path, file2_path):
    """
    Compare line endings of two text files and analyze punctuation patterns.
    Returns statistics about matching and mismatching line endings.
    
    Args:
        file1_path (str): Path to first text file
        file2_path (str): Path to second text file
    """
    # Define end punctuation marks
    end_punct = {'.', '!', '?'}
    
    # Statistics counters
    stats = {
        'total_lines': 0,
        'matching_endings': 0,
        'both_with_punct': 0,
        'both_without_punct': 0,
        'only_file1_punct': 0,
        'only_file2_punct': 0
    }
    
    def has_end_punct(line):
        """Check if line ends with punctuation, handling whitespace."""
        line = line.rstrip()
        return line[-1] in end_punct if line else False
    
    try:
        with open(file1_path, 'r', encoding='utf-8') as f1, \
             open(file2_path, 'r', encoding='utf-8') as f2:
            
            for line1, line2 in zip(f1, f2):
                stats['total_lines'] += 1
                
                # Check punctuation in both lines
                punct1 = has_end_punct(line1)
                punct2 = has_end_punct(line2)
                
                # Update statistics
                if punct1 == punct2:
                    stats['matching_endings'] += 1
                    if punct1:
                        stats['both_with_punct'] += 1
                    else:
                        stats['both_without_punct'] += 1
                else:
                    if punct1:
                        stats['only_file1_punct'] += 1
                    else:
                        stats['only_file2_punct'] += 1
                        
        # Calculate percentages
        if stats['total_lines'] > 0:
            stats['matching_percentage'] = (stats['matching_endings'] / stats['total_lines']) * 100
            
        return stats
        
    except FileNotFoundError as e:
        return f"Error: {e}"
    except UnicodeDecodeError:
        return "Error: Unable to decode file. Please ensure it's in UTF-8 format."

# Example usage
if __name__ == "__main__":
    # Replace with your file paths
    file1 = "text1.txt"
    file2 = "text2.txt"
    
    results = analyze_line_endings(file1, file2)
    
    if isinstance(results, dict):
        print(f"Total lines analyzed: {results['total_lines']}")
        print(f"Lines with matching endings: {results['matching_endings']} ({results['matching_percentage']:.2f}%)")
        print(f"Lines both with punctuation: {results['both_with_punct']}")
        print(f"Lines both without punctuation: {results['both_without_punct']}")
        print(f"Lines with punctuation only in file 1: {results['only_file1_punct']}")
        print(f"Lines with punctuation only in file 2: {results['only_file2_punct']}")
    else:
        print(results)  # Print error message