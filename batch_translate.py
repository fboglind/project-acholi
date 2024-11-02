import argparse
import os
import subprocess
import logging
from sacrebleu.metrics import BLEU, CHRF
from subword_nmt.apply_bpe import BPE
from typing import Dict, List
import pandas as pd
from datetime import datetime

class BatchTranslator:
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.model_dir = os.path.join(project_dir, "onmt_data/onmt_model")
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics
        self.bleu = BLEU()
        self.chrf = CHRF()
        
        # Results storage
        self.results = []

    def get_checkpoints(self) -> List[str]:
        """Get all model checkpoints"""
        checkpoints = [f for f in os.listdir(self.model_dir) 
                      if f.startswith("model_step_") and f.endswith(".pt")]
        return sorted(checkpoints, key=lambda x: int(x.split("_")[2].split(".")[0]))

    def apply_bpe(self, test_file: str, bpe_codes: str, output_file: str):
        """Apply BPE encoding to test data"""
        with open(bpe_codes, 'r', encoding='utf-8') as codes_file:
            bpe = BPE(codes_file)
            
        with open(test_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                encoded_line = bpe.process_line(line.strip())
                outfile.write(encoded_line + '\n')

    def translate(self, 
                 checkpoint: str, 
                 src_file: str, 
                 output_file: str,
                 beam_size: int = 5,
                 batch_size: int = 32) -> str:
        """Run translation with specific checkpoint and parameters"""
        model_path = os.path.join(self.model_dir, checkpoint)
        
        cmd = [
            "onmt_translate",
            "-model", model_path,
            "-src", src_file,
            "-output", output_file,
            "-gpu", "0",
            "-batch_size", str(batch_size),
            "-beam_size", str(beam_size),
            "-replace_unk"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return output_file
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Translation failed for {checkpoint}: {e}")
            return None

    def remove_bpe(self, file_path: str) -> str:
        """Remove BPE tokens from translated output"""
        output_path = file_path.replace('.bpe.', '.')
        with open(file_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                line = line.replace('@@ ', '')
                outfile.write(line)
        return output_path

    def evaluate(self, 
                hypothesis_file: str, 
                reference_file: str) -> Dict[str, float]:
        """Calculate BLEU and chrF scores"""
        with open(hypothesis_file, 'r', encoding='utf-8') as f:
            hypotheses = [line.strip() for line in f]
        with open(reference_file, 'r', encoding='utf-8') as f:
            references = [line.strip() for line in f]
            
        bleu_score = self.bleu.corpus_score(hypotheses, [references])
        chrf_score = self.chrf.corpus_score(hypotheses, [references])
        
        return {
            'bleu': bleu_score.score,
            'chrf': chrf_score.score
        }

    def run_batch_translation(self, 
                            test_src: str,
                            test_ref: str,
                            bpe_codes: str,
                            beam_sizes: List[int] = [5],
                            batch_sizes: List[int] = [32]):
        """Run translations with different checkpoints and parameters"""
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.project_dir, f"translations_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Apply BPE to test data
        bpe_test = os.path.join(output_dir, "test.bpe.ach")
        self.apply_bpe(test_src, bpe_codes, bpe_test)
        
        # Get all checkpoints
        checkpoints = self.get_checkpoints()
        
        # Run translations with different parameters
        for checkpoint in checkpoints:
            for beam_size in beam_sizes:
                for batch_size in batch_sizes:
                    self.logger.info(f"Translating with checkpoint {checkpoint}, "
                                   f"beam_size={beam_size}, batch_size={batch_size}")
                    
                    # Generate output path
                    output_base = f"trans_step{checkpoint.split('_')[2].split('.')[0]}_beam{beam_size}_batch{batch_size}"
                    output_bpe = os.path.join(output_dir, f"{output_base}.bpe.txt")
                    
                    # Translate
                    if self.translate(checkpoint, bpe_test, output_bpe, beam_size, batch_size):
                        # Remove BPE
                        output_clean = self.remove_bpe(output_bpe)
                        
                        # Evaluate
                        scores = self.evaluate(output_clean, test_ref)
                        
                        # Store results
                        result = {
                            'checkpoint': checkpoint,
                            'step': int(checkpoint.split('_')[2].split('.')[0]),
                            'beam_size': beam_size,
                            'batch_size': batch_size,
                            'bleu': scores['bleu'],
                            'chrf': scores['chrf'],
                            'output_file': output_clean
                        }
                        self.results.append(result)
        
        # Create results summary
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('bleu', ascending=False)
        
        # Save results
        results_file = os.path.join(output_dir, "translation_results.csv")
        results_df.to_csv(results_file, index=False)
        
        # Print best results
        best_result = results_df.iloc[0]
        self.logger.info("\nBest configuration:")
        self.logger.info(f"Checkpoint: {best_result['checkpoint']}")
        self.logger.info(f"Step: {best_result['step']}")
        self.logger.info(f"Beam size: {best_result['beam_size']}")
        self.logger.info(f"Batch size: {best_result['batch_size']}")
        self.logger.info(f"BLEU score: {best_result['bleu']:.2f}")
        self.logger.info(f"chrF score: {best_result['chrf']:.2f}")
        self.logger.info(f"Output file: {best_result['output_file']}")
        
        return results_df

def main():
    parser = argparse.ArgumentParser(description='Batch translate and evaluate MT models')
    parser.add_argument('--project-dir', required=True,
                       help='Project root directory')
    parser.add_argument('--test-src', required=True,
                       help='Source language test file')
    parser.add_argument('--test-ref', required=True,
                       help='Reference translations file')
    parser.add_argument('--bpe-codes', required=True,
                       help='BPE codes file')
    parser.add_argument('--beam-sizes', type=int, nargs='+', default=[5],
                       help='Beam sizes to try')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[32],
                       help='Batch sizes to try')
    
    args = parser.parse_args()
    
    translator = BatchTranslator(args.project_dir)
    results = translator.run_batch_translation(
        args.test_src,
        args.test_ref,
        args.bpe_codes,
        args.beam_sizes,
        args.batch_sizes
    )

if __name__ == "__main__":
    main()

## Command:
# python batch_translate.py \
#     --project-dir /proj/uppmax2024-2-13/private/acholi_mt24/project-acholi-mt24 \
#     --test-src processed_data_moses/salt.test.tk.lc.ach \
#     --test-ref processed_data_moses/salt.test.tk.lc.eng \
#     --bpe-codes onmt_data/data.ach.codes \
#     --beam-sizes 3 5 7 \
#     --batch-sizes 16 32 64