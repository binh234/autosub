from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import kenlm
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel, build_ctcdecoder
import torch
import time
import tqdm
from multiprocessing import Pool
import warnings

from audio import AudioFile

WINDOW_SIZE = 25
STRIDES = 20

# warnings.filterwarnings("error")


class ASRModel:
    def __init__(self, model_path, lm_path=None, hot_words=[]):
        super(ASRModel, self).__init__()
        self.model_path = model_path
        self.lm_path = lm_path
        self.hot_words = hot_words

        print("Loading model...")
        start = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path).to(self.device)
        print("Model loaded successfully in %fs" % (time.time() - start))
        
        # Sanity check
        x = torch.zeros([1, 10000])
        with torch.no_grad():
            out = self.model(x).logits
            self.vocab_size = out.shape[-1]

        self.decoder = self.build_lm(self.processor.tokenizer)
        print("Language model loaded successfully in %fs" %
              (time.time() - start))


    def build_lm(self, tokenizer):
        vocab_dict = tokenizer.get_vocab()
        sort_vocab = sorted((value, key)
                            for (key, value) in vocab_dict.items())
        vocab = [x[1] for x in sort_vocab][:self.vocab_size]
        vocab_list = vocab
        # convert ctc blank character representation
        vocab_list[tokenizer.pad_token_id] = ""
        # # replace special characters
        # vocab_list[tokenizer.unk_token_id] = ""
        # vocab_list[tokenizer.bos_token_id] = ""
        # vocab_list[tokenizer.eos_token_id] = ""
        # convert space character representation
        vocab_list[tokenizer.word_delimiter_token_id] = " "
        # specify ctc blank char index, since conventially it is the last entry of the logit matrix
        # alphabet = Alphabet.build_alphabet(
        #     vocab_list, ctc_token_idx=tokenizer.pad_token_id)

        # if self.lm_path is None:
        #     lm_model = None
        # else:
        #     lm_model = LanguageModel(kenlm.Model(self.lm_path))
        decoder = build_ctcdecoder(vocab_list, self.lm_path)
        return decoder

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

        return [{'start': start,
                 'end': end,
                 'transcript': transcript,
                 'tokens': tokens,
                 'score': score} for start, end, transcript, tokens, score in result]

    def transcribe(self, audio):
        if len(audio.shape) == 1:
            audio = audio
        elif len(audio.shape) > 2:
            raise ValueError(
                "Expected 2 dimensions input, but got %d dimensions" % len(audio.shape))

        inputs = self.processor(
            [x for x in audio], sampling_rate=16_000, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            attention_mask = None if not hasattr(inputs, 'attention_mask') else inputs.attention_mask
            logits = self.model(inputs.input_values,
                                attention_mask=attention_mask,
                                ).logits

        with Pool() as pool:
            pred_str = self.decoder.decode_batch(pool=pool, logits_list=logits.cpu().detach().numpy(), beam_width=100)

        return pred_str

    def transcribe_with_metadata(self, audio, start):
        if len(audio.shape) == 1:
            audio = audio
        elif len(audio.shape) > 2:
            raise ValueError(
                "Expected 2 dimensions input, but got %d dimensions" % len(audio.shape))

        inputs = self.processor(
            [x for x in audio], sampling_rate=16_000, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            attention_mask = None if not hasattr(inputs, 'attention_mask') else inputs.attention_mask
            logits = self.model(inputs.input_values,
                                attention_mask=attention_mask,
                                ).logits

        with Pool() as pool:
            beam_batch = self.decoder.decode_beams_batch(pool=pool, logits_list=logits.cpu().detach().numpy(), beam_width=100)

        pred_batch = []
        for top_beam in beam_batch:
            beam = top_beam[0]
            tokens = []
            score = beam[3]

            for w, i in beam[1]:
                tokens.append({
                    'text': w,
                    'start': start + i[0] * STRIDES,
                    'end': start + i[1] * STRIDES + WINDOW_SIZE,
                })

            pred_batch.append((beam[0], tokens, score))

        return pred_batch
