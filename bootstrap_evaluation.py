"""bootstrap_evaluation.py contains code for performing bootstrap evaluation of machine translation output"""
import numpy as np
import statistics
import time
from datetime import datetime
import subprocess
import os
import shutil
from pathlib import Path
import tempfile
from evaluation import eval

class OpenNMTBootstrapEvaluator:
    def __init__(self, 
                 src_file: str,
                 baseline_model_path: str,
                 experimental_model_path: str,
                 ref_file: str,
                 eval_class,
                 n_iterations: int = 1000,
                 batch_size: int = 32,
                 beam_size: int = 5,
                 gpu: str = "0",
                 temp_dir: str = "bootstrap_temp") -> None:
        """
        Initialize the bootstrap evaluator for comparing two OpenNMT models.
        
        Args:
            src_file: Path to source text file
            baseline_model_path: Path to baseline OpenNMT model
            experimental_model_path: Path to experimental OpenNMT model
            ref_file: Path to reference translations
            eval_class: The evaluation class to use
            n_iterations: Number of bootstrap iterations
            batch_size: Batch size for OpenNMT translation
            beam_size: Beam size for OpenNMT translation
            gpu: GPU device to use (e.g., "0" or "-1" for CPU)
            temp_dir: Directory for temporary files
        """
        self.src_file = src_file
        self.baseline_model = baseline_model_path
        self.experimental_model = experimental_model_path
        self.ref_file = ref_file
        self.eval_class = eval_class
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.gpu = gpu
        
        # Create temp directory if it doesn't exist
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Load source and reference data
        self.src_lines = self._read_file(src_file)
        self.ref_lines = self._read_file(ref_file)
        
        # Validate data
        assert len(self.src_lines) == len(self.ref_lines), \
            "Source and reference files must have the same number of lines"
            
        self.n_sentences = len(self.src_lines)

    def _read_file(self, filepath: str) -> list[str]:
        """Read lines from a file and strip whitespace."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]

    def _write_temp_file(self, lines: list[str], filepath: str) -> None:
        """Write lines to a temporary file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')

    def _translate_with_onmt(self, model_path: str, src_file: str, output_file: str) -> None:
        """Run OpenNMT translation."""
        cmd = [
            "onmt_translate",
            "-model", model_path,
            "-src", src_file,
            "-output", output_file,
            "-gpu", self.gpu,
            "-batch_size", str(self.batch_size),
            "-beam_size", str(self.beam_size),
            "-replace_unk"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Translation error: {e.stderr}")
            raise

    def evaluate_models_on_sample(self, sample_indices: list[int]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """
        Evaluate both models on a bootstrap sample.
        Returns tuples of (BLEU, METEOR, COMET) scores for both systems.
        """
        # Create temporary files for this sample
        iteration_id = int(time.time() * 1000)  # Use timestamp for unique files
        temp_src = self.temp_dir / f"src_{iteration_id}.txt"
        temp_base_out = self.temp_dir / f"base_out_{iteration_id}.txt"
        temp_exp_out = self.temp_dir / f"exp_out_{iteration_id}.txt"
        temp_ref = self.temp_dir / f"ref_{iteration_id}.txt"

        try:
            # Write sampled sentences to temporary files
            sampled_src = [self.src_lines[i] for i in sample_indices]
            sampled_ref = [self.ref_lines[i] for i in sample_indices]
            
            self._write_temp_file(sampled_src, temp_src)
            self._write_temp_file(sampled_ref, temp_ref)

            # Generate translations with both models
            self._translate_with_onmt(self.baseline_model, temp_src, temp_base_out)
            self._translate_with_onmt(self.experimental_model, temp_src, temp_exp_out)

            # Evaluate baseline system
            base_eval = self.eval_class(str(temp_src), str(temp_base_out), str(temp_ref))
            base_eval.full_evaluation()
            base_scores = (base_eval.bleu_score, base_eval.meteor_score, base_eval.comet_score)

            # Evaluate experimental system
            exp_eval = self.eval_class(str(temp_src), str(temp_exp_out), str(temp_ref))
            exp_eval.full_evaluation()
            exp_scores = (exp_eval.bleu_score, exp_eval.meteor_score, exp_eval.comet_score)

            return base_scores, exp_scores

        finally:
            # Clean up temporary files
            for file in [temp_src, temp_base_out, temp_exp_out, temp_ref]:
                if file.exists():
                    file.unlink()

    def run_bootstrap(self) -> dict:
        """
        Run bootstrap resampling evaluation.
        Returns a dictionary with detailed results and statistics.
        """
        print(f"Starting bootstrap evaluation with {self.n_iterations} iterations...")
        start_time = time.time()
        
        # Initialize result storage
        wins = {'bleu': {'baseline': 0, 'experimental': 0, 'tie': 0},
                'meteor': {'baseline': 0, 'experimental': 0, 'tie': 0},
                'comet': {'baseline': 0, 'experimental': 0, 'tie': 0}}
        
        scores = {'baseline': {'bleu': [], 'meteor': [], 'comet': []},
                 'experimental': {'bleu': [], 'meteor': [], 'comet': []}}
        
        try:
            # Run bootstrap iterations
            for i in range(self.n_iterations):
                if (i + 1) % 10 == 0:
                    print(f"Completed {i + 1} iterations...")
                
                # Generate bootstrap sample indices
                indices = np.random.choice(self.n_sentences, size=self.n_sentences, replace=True)
                
                # Evaluate both systems
                base_scores, exp_scores = self.evaluate_models_on_sample(indices)
                
                # Record scores
                for metric_idx, metric in enumerate(['bleu', 'meteor', 'comet']):
                    scores['baseline'][metric].append(base_scores[metric_idx])
                    scores['experimental'][metric].append(exp_scores[metric_idx])
                    
                    # Count wins
                    if base_scores[metric_idx] > exp_scores[metric_idx]:
                        wins[metric]['baseline'] += 1
                    elif exp_scores[metric_idx] > base_scores[metric_idx]:
                        wins[metric]['experimental'] += 1
                    else:
                        wins[metric]['tie'] += 1

            # Calculate final statistics
            results = {
                'iterations': self.n_iterations,
                'total_sentences': self.n_sentences,
                'time_taken': time.time() - start_time,
                'metrics': {}
            }
            
            for metric in ['bleu', 'meteor', 'comet']:
                results['metrics'][metric] = {
                    'baseline_mean': statistics.mean(scores['baseline'][metric]),
                    'experimental_mean': statistics.mean(scores['experimental'][metric]),
                    'baseline_std': statistics.stdev(scores['baseline'][metric]),
                    'experimental_std': statistics.stdev(scores['experimental'][metric]),
                    'baseline_wins': wins[metric]['baseline'],
                    'experimental_wins': wins[metric]['experimental'],
                    'ties': wins[metric]['tie'],
                    'experimental_win_ratio': wins[metric]['experimental'] / self.n_iterations,
                    'p_value': (wins[metric]['baseline'] + wins[metric]['tie'] / 2) / self.n_iterations
                }
            
            return results

        finally:
            # Clean up temp directory
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)

    def print_results(self, results: dict) -> None:
        """Print formatted results of the bootstrap evaluation."""
        print("\n=== Bootstrap Evaluation Results ===")
        print(f"Number of iterations: {results['iterations']}")
        print(f"Total sentences: {results['total_sentences']}")
        print(f"Time taken: {results['time_taken']:.2f} seconds\n")
        
        for metric, stats in results['metrics'].items():
            print(f"\n{metric.upper()} Scores:")
            print(f"Baseline: {stats['baseline_mean']:.4f} (±{stats['baseline_std']:.4f})")
            print(f"Experimental: {stats['experimental_mean']:.4f} (±{stats['experimental_std']:.4f})")
            print(f"Wins: Baseline: {stats['baseline_wins']}, "
                  f"Experimental: {stats['experimental_wins']}, "
                  f"Ties: {stats['ties']}")
            print(f"Experimental win ratio: {stats['experimental_win_ratio']:.4f}")
            print(f"Approximate p-value: {stats['p_value']:.4f}")


# Enter correct paths
evaluator = OpenNMTBootstrapEvaluator(
    src_file="processed_data_moses/salt.test.tk.lc.ach",
    baseline_model_path="onmt_data/onmt_model",
    experimental_model_path="experimental_model",
    ref_file="processed_data_moses/salt.test.tk.lc.eng",
    eval_class=eval,  
    n_iterations=1000,
    batch_size=32,
    beam_size=5,
    gpu="0"
)

# Run the bootstrap evaluation
results = evaluator.run_bootstrap()

# Print the results
evaluator.print_results(results)

if __name__ == "__main__":
    main()