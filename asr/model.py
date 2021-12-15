from speechbrain.pretrained import EncoderASR
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel, build_ctcdecoder
import torch
import numpy as np
import time
import tqdm
from multiprocessing import Pool
import warnings

from asr.audio import AudioFile

WINDOW_SIZE = 25
STRIDES = 20

# warnings.filterwarnings("error")


class ASRModel:
    def __init__(self, model_path, lm_path=None, vocab_path=None, hot_words=[]):
        super(ASRModel, self).__init__()
        self.model_path = model_path
        self.lm_path = lm_path
        self.vocab_path = vocab_path
        self.hot_words = hot_words

        print("Loading model...")
        start = time.time()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path).to(self.device)
        print("Model loaded successfully in %fs" % (time.time() - start))

        # Sanity check
        x = torch.zeros([1, 4000]).to(self.device)
        with torch.no_grad():
            out = self.model(x).logits
            self.vocab_size = out.shape[-1]

        self.decoder = self.build_lm(self.processor.tokenizer, self.vocab_path)
        print("Language model loaded successfully in %fs" %
              (time.time() - start))

    def build_lm(self, tokenizer, vocab_path=None):
        unigrams = None
        if vocab_path is not None:
            unigrams = []
            with open(vocab_path, encoding='utf-8') as f:
                for line in f:
                    unigrams.append(line.strip())

        vocab_dict = tokenizer.get_vocab()
        sort_vocab = sorted((value, key)
                            for (key, value) in vocab_dict.items())
        vocab = [x[1] for x in sort_vocab][:self.vocab_size]

        vocab_list = vocab
        # convert ctc blank character representation
        vocab_list[tokenizer.pad_token_id] = ""
        # replace special characters
        vocab_list[tokenizer.word_delimiter_token_id] = " "
        # specify ctc blank char index, since conventionally it is the last entry of the logit matrix
        decoder = build_ctcdecoder(vocab_list, self.lm_path, unigrams=unigrams)
        return decoder
    
    def predict(self, audio):
        if len(audio.shape) == 1:
            audio = [audio]
        elif len(audio.shape) > 2:
            raise ValueError(
                "Expected 2 dimensions input, but got %d dimensions" % len(audio.shape))

        inputs = self.processor(
            list(audio), sampling_rate=16_000, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            attention_mask = None if not hasattr(
                inputs, 'attention_mask') else inputs.attention_mask
            logits = self.model(inputs.input_values,
                                attention_mask=attention_mask,
                                ).logits
        
        return logits

    def transcribe_file(self, audio_path):
        audio_file = AudioFile(audio_path)
        progress_bar = tqdm.tqdm(total=int(audio_file.audio_length * 1000))
        last = 0
        result = []
        for start, end, audio in audio_file.split():
            transcript = self.transcribe(audio)[0]
            result.append((start, end, transcript.strip()))

            progress_bar.update(int(end - last))
            last = end

        audio_file.close()

        return [{'start': start,
                 'end': end,
                 'transcript': transcript} for start, end, transcript in result]

    def transcribe_file_with_metadata(self, audio_path):
        audio_file = AudioFile(audio_path)
        progress_bar = tqdm.tqdm(total=int(audio_file.audio_length * 1000))
        last = 0
        result = []
        for start, end, audio in audio_file.split():
            transcript, tokens, score = self.transcribe_with_metadata(audio, start)[0]
            result.append((start, end, transcript.strip(), tokens, score))

            progress_bar.update(int(end - last))
            last = end

        audio_file.close()

        return [{'start': start,
                 'end': end,
                 'transcript': transcript,
                 'tokens': tokens,
                 'score': score} for start, end, transcript, tokens, score in result]

    def transcribe(self, audio):
        logits = self.predict(audio)

        pred_str = [self.decoder.decode(
            logit.detach().cpu().numpy(), beam_width=100) for logit in logits]

        return pred_str

    def transcribe_with_metadata(self, audio, start):
        logits = self.predict(audio)

        beam_batch = [self.decoder.decode_beams(
            logit.detach().cpu().numpy(), beam_width=100) for logit in logits]
        pred_batch = []
        for top_beam in beam_batch:
            beam = top_beam[0]
            tokens = []
            score = beam[3]

            for w, i in beam[2]:
                tokens.append({
                    'text': w,
                    'start': start + i[0] * STRIDES,
                    'end': start + i[1] * STRIDES + WINDOW_SIZE,
                })

            pred_batch.append((beam[0], tokens, score))

        return pred_batch

class EncoderASRModel(ASRModel):
    def __init__(self, model_path, lm_path=None, vocab_path=None, hot_words=[]):
        # super(EncoderASRModel, self).__init__(model_path, lm_path, vocab_path, hot_words)
        self.model_path = model_path
        self.lm_path = lm_path
        self.vocab_path = vocab_path
        self.hot_words = hot_words

        print("Loading model...")
        start = time.time()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = EncoderASR.from_hparams(
            source="dragonSwing/wav2vec2-base-vn-270h", 
            savedir="pretrain/asr_model", 
            run_opts={'device': self.device}
        )
        print("Model loaded successfully in %fs" % (time.time() - start))

        # Sanity check
        x = torch.zeros([1, 4000]).to(self.device)
        with torch.no_grad():
            out = self.model(x, torch.ones(1))
            self.vocab_size = out.shape[-1]

        self.decoder = self.build_lm(self.model.tokenizer, self.vocab_path)
        print("Language model loaded successfully in %fs" %
              (time.time() - start))
    
    def predict(self, audio):
        if len(audio.shape) == 1:
            audio = [audio]
        elif len(audio.shape) > 2:
            raise ValueError(
                "Expected 2 dimensions input, but got %d dimensions" % len(audio.shape))

        inputs = torch.FloatTensor(audio)
        input_lens = torch.ones(inputs.shape[0])
        with torch.no_grad():
            logits = self.model(inputs, input_lens)
        
        return logits