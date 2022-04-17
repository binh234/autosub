import os
import torch
import warnings
from speechbrain.pretrained import EncoderClassifier
from utils import DEFAULT_TEMP_DIR

warnings.filterwarnings('ignore')

# _DEFAULT_CHECKPOINT = '/content/drive/MyDrive/audify/inference'
_DEFAULT_CHECKPOINT = 'dragonSwing/audify'
_DEFAULT_SAVE_PATH = os.path.join(DEFAULT_TEMP_DIR, 'audify')

class AudifyModel(torch.nn.Module):
    instance_dict = {}
    def __init__(self, model_path, save_path):
        super(AudifyModel, self).__init__()
        self.model_path = model_path
        self.save_path = save_path
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.classifier = EncoderClassifier.from_hparams(
            source=model_path, 
            savedir=save_path,
            run_opts={"device": self.device}
        )
    
    @classmethod
    def get_instance(cls, model_path=_DEFAULT_CHECKPOINT, save_path=_DEFAULT_SAVE_PATH):
        if model_path not in cls.instance_dict:
            cls.instance_dict[model_path] = AudifyModel(model_path, save_path)
        return cls.instance_dict[model_path]
    
    def forward(self, wavs, wav_lens=None):
        """Runs the classification"""
        return self.classifier(wavs, wav_lens)

    def classify_file(self, file_path):
        out_prob, score, index, text_lab = self.classifier.classify_file(file_path)

        return out_prob, score, index, text_lab
