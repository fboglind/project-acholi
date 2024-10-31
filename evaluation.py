import nltk.translate.meteor_score as meteor
# import nltk.translate.ribes_score as ribes
import nltk.translate.bleu_score as bleu
from comet import download_model, load_from_checkpoint

class eval:
    def __init__(self, model_path="Unbabel/XCOMET-XL") -> None:
        
        self.model_path = download_model(model_path)
        self.comet_model = load_from_checkpoint(self.model_path)
    
    def bleu(self, hypothesis: list[str], refferences: list[list[str]]):
        
        return bleu.sentence_bleu(references=refferences, hypothesis=hypothesis)
    
    def meteor (self,  hypothesis: list[str], refferences: list[list[str]]):
        
        return meteor.meteor_score(hypothesis=hypothesis, references=refferences)
    
    def comet(self, src: str, hyp: str, ref: str):
        data = [{"src": src, "mt": hyp, "ref": ref}]

        model_output = self.comet_model.predict(samples=data)
        return model_output.scores()

if __name__ =="__main__":
    pass 