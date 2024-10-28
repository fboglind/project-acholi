from rouge_metric import PyRouge
import nltk.translate.nist_score as nist
# import nltk.translate.meteor_score as meteor
# import nltk.translate.ribes_score as ribes
# import nltk.translate.bleu_score as bleu

class eval:
    def __init__(self) -> None:
        pass
    
    def nist(self, hypothesis: list[str], refferences: list[list[str]]):
        #TODO implementera nist evaluation
        return nist.sentence_nist(references=refferences, hypothesis=hypothesis)
    
    def rouge(self):
        #TODO implementer rouge evaluation
        pass

if __name__ =="__main__":
    pass 