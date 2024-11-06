import nltk.translate.meteor_score as meteor
# import nltk.translate.ribes_score as ribes
import nltk.translate.bleu_score as bleu
import nltk
from comet import download_model, load_from_checkpoint
import statistics

#Unbabel/XCOMET-XL
#Unbabel/wmt22-comet-da
class eval:
    def __init__(self, source_file, translation_out, refference_file, model) -> None:
        """this class contains methods to evaluate the quality of a machine translation. to use it, pass three files of paralell translation
        one file conatins the untranslated source text, one is a reliable paralell translation of the source file, 
        and one contains a machine traslated attempt att translating the source file
        make sure that each line of each file is a paralell match to each line of every other file.

        the class employs BLEU, meteor, and comet translation methods. 
        
        Args:
            source_file (str): path to a soruce file from which the translation is done
            translation_out (str): path to a file where a output of a translation model is written
            refference_file (str): path to a flie where the paralell refference translation of the source
            model (str, optional): a comet model aquired with comet.download_model(["model name"]). Normally "Unbabel/wmt20-comet-qe-da". 
            others include "Unbabel/wmt22-comet-da" https://huggingface.co/Unbabel for more.
        """
        
        self.bleu_score = float
        self.comet_score = float
        self.comet_score_list = []
        self.meteor_score = float
        self.meteor_score_list = []

        self.src = source_file
        self.trans = translation_out
        self.ref = refference_file
        self.model_path = model
        self.comet_model = load_from_checkpoint(self.model_path)
        
        nltk.download('wordnet')

    def bleu(self, hypothesis: list[list[str]], refferences: list[list[str]]) -> float:
        """calculates BLEU-score for all lines

        Args:
            hypothesis (list[list[str]]): list of machine translated tokenised line 
            refferences (list[list[str]]): list of tokenized reference line. 

        Returns:
            BLEU-score(float): combined score of all translated line. 
        """
        return bleu.corpus_bleu(list_of_references=refferences, hypotheses=hypothesis)
    
    def meteor (self,  hypothesis: list[str], refferences: list[list[str]]) -> list[float]:

        """calulates meteor score for one line
        Args: 
            hypothesis (list[str]):  machine translated tokenised line 
            refferences (list[list[str]]): list of tokenized reference line.
        Returns:
            meteor score (float): list of score for one line
        """
        
        return meteor.meteor_score(references=refferences, hypothesis=hypothesis)
    
    def comet(self, data: list[dict]) -> list[float]: 
        """calculates comet score for all lines

        Args:
            data list[dict]): dictionary of source, machine translation and refference (src, mt, ref respecitvely) must be in the followint format:
            [{
                "src": "10 到 15 分钟可以送到吗",
                "mt": "Can I receive my food in 10 to 15 minutes?",
                "ref": "Can it be delivered between 10 to 15 minutes?"
            },
            {
                "src": "Pode ser entregue dentro de 10 a 15 minutos?",
                "mt": "Can you send it for 10 to 15 minutes?",
                "ref": "Can it be delivered between 10 to 15 minutes?"
            }]

        Returns:
            list(float)
        """

        model_output = self.comet_model.predict(samples=data)
        return model_output.scores
    
    def full_evaluation(self, do_you_want_to_run_comet=True):
        """makes a full evaluation of the translation using METEOR, COMET and BLEU
        """
        src = []
        hypothesis = []
        bleu_refference = []
        refference = []
        with open (self.src) as f:
            for line in f:
                src.append(line)
        with open (self.trans) as g:
            for line in g:
                hypothesis.append(line.split())
        with open (self.ref) as h:
            for line in h:
                bleu_refference.append([line.split()])
                refference.append(line.split())
        
        
        for hyp, ref in zip (hypothesis, refference):       
            self.meteor_score_list.append(self.meteor(refferences=[ref], hypothesis=hyp))
        if do_you_want_to_run_comet:
            print("running comet")
            comet_data = []
            for src, hyp, ref in zip (src, hypothesis, refference):
                
                comet_data.append({"src":src, "mt": " ".join(hyp), "ref": " ".join(ref)})
        
        if do_you_want_to_run_comet:    
            self.comet_score_list = (self.comet(comet_data))
            self.comet_score = statistics.mean(self.comet_score_list)
        
        self.meteor_score = statistics.mean(self.meteor_score_list)
        self.bleu_score = self.bleu(hypothesis=hypothesis, refferences=bleu_refference)
        
        # printing out the resiults for now. make better output later.       
        print ("COMET score: ", self.comet_score)
        print ("METEOR score: ", self.meteor_score)
        print ("BLEU score: ",  self.bleu_score)




if __name__ =="__main__":

    # so far, this is the interface. make an argparse later
    source_file = "processed_data_moses/salt.test.tk.lc.ach"
    translation_out = "translations_20241102_175428/trans_step8000_beam5_batch32.txt"
    refference_file  = "processed_data_moses/salt.test.tk.lc.eng"
    
    comet_model = download_model("Unbabel/wmt20-comet-qe-da")
    ev = eval(source_file, translation_out, refference_file, comet_model)
   
    
    ev.full_evaluation()
   