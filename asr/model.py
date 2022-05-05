import functools
import operator
from speechbrain.pretrained import EncoderASR
from transformers import AutoModelForCTC, AutoProcessor
from pyctcdecode import build_ctcdecoder
import torch
import numpy as np
import time
import tqdm
import warnings

from asr.audio import AudioFile

WINDOW_SIZE = 25
STRIDES = 20

# warnings.filterwarnings("error")


class BaseASRModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseASRModel, self).__init__()

    def build_lm(self, tokenizer, vocab_path=None):
        unigrams = None
        if vocab_path is not None:
            unigrams = []
            with open(vocab_path, encoding='utf-8') as f:
                for line in f:
                    unigrams.append(line.strip())

        vocab_dict = tokenizer.get_vocab()
        sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
        vocab = [x[1] for x in sort_vocab][: self.vocab_size]

        vocab_list = vocab
        # convert ctc blank character representation
        vocab_list[tokenizer.pad_token_id] = ""
        # replace special characters
        vocab_list[tokenizer.word_delimiter_token_id] = " "
        # specify ctc blank char index, since conventionally it is the last entry of the logit matrix
        decoder = build_ctcdecoder(vocab_list, self.lm_path, unigrams=unigrams)
        return decoder

    def forward(self, audio):
        raise NotImplementedError("")

    def forward_streaming(self, audio):
        raise NotImplementedError("")

    def _split_chunks(self, batch):
        chunk_size = self.chunk_size
        context_size = self.context_size
        overlap_size = context_size * 2
        stride = chunk_size - overlap_size
        result = []
        num_sample = batch.size(1)

        if num_sample <= chunk_size:
            result.append(batch)
        elif num_sample > chunk_size and num_sample <= (chunk_size * 2 - overlap_size):
            split_idx = ((num_sample + overlap_size + STRIDES) // (2 * STRIDES)) * STRIDES
            result.append(batch[:, :split_idx])
            result.append(batch[:, split_idx - overlap_size :])
        else:
            for i in range(0, num_sample - overlap_size, stride):
                result.append(batch[:, i : i + chunk_size])

        return result

    def _merge_chunks(self, batch):
        result = []
        context_size = self.context_size
        context_tokens = self.context_tokens
        num_chunks = len(batch)

        if len(batch) == 1 or context_size == 0:
            for sub_logits in batch:
                result.append(sub_logits)
        else:
            for i, sub_logits in enumerate(batch):
                if i == 0:
                    result.append(sub_logits[:, :-context_tokens])
                elif i == num_chunks - 1:
                    result.append(sub_logits[:, context_tokens - 1 :])
                else:
                    result.append(sub_logits[:, context_tokens - 1 : -context_tokens])

        logits = torch.cat(result, dim=1)
        return logits

    def transcribe_file(self, audio_path):
        allow_tags = {"speech", "male", "female", "noisy_speech"}
        audio_file = AudioFile(audio_path)
        progress_bar = tqdm.tqdm(total=int(audio_file.audio_length * 1000))
        last = 0
        result = []
        for (start, end, audio, tag) in audio_file.split():
            if tag not in allow_tags:
                pass
            transcript = self.transcribe(audio)[0]
            result.append((start, end, transcript.strip()))

            progress_bar.update(int(end - last))
            last = end

        audio_file.close()

        return [{'start': start, 'end': end, 'transcript': transcript} for start, end, transcript in result]

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

        return [
            {'start': start, 'end': end, 'transcript': transcript, 'tokens': tokens, 'score': score}
            for start, end, transcript, tokens, score in result
        ]

    def transcribe(self, audio, beam_width=100):
        logits = self.forward_streaming(audio)

        pred_str = [
            self.decoder.decode(logit.detach().cpu().numpy(), beam_width=beam_width, hotwords=self.hot_words)
            for logit in logits
        ]

        return pred_str

    def transcribe_with_metadata(self, audio, start=0, beam_width=100):
        logits = self.forward_streaming(audio)

        beam_batch = [
            self.decoder.decode_beams(logit.detach().cpu().numpy(), beam_width=beam_width, hotwords=self.hot_words)
            for logit in logits
        ]
        pred_batch = []
        for top_beam in beam_batch:
            beam = top_beam[0]
            tokens = []
            score = beam[3]

            for w, i in beam[2]:
                tokens.append(
                    {
                        'text': w,
                        'start': start + i[0] * STRIDES,
                        'end': start + i[1] * STRIDES + WINDOW_SIZE,
                    }
                )

            pred_batch.append((beam[0], tokens, score))

        return pred_batch


class HuggingFaceASRModel(BaseASRModel):
    def __init__(
        self,
        model_path,
        lm_path=None,
        vocab_path=None,
        cache_dir=None,
        sampling_rate=16_000,
        chunk_size=20,
        context_size=2.5,
        hot_words=[],
        **kwargs,
    ):
        r"""
        Args:
            model_path (`Union[str, os.PathLike]`):
                Path to pretrained model.
            lm_path (`Union[str, os.PathLike]`, *Optional*):
                Path to language model file.
            vocab_path (`Union[str, os.PathLike]`, *Optional*):
                Path to vocabulary file.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            sampling_rate (`int`, defaults to 16000):
                Sampling rate of audio inputs.
            chunk_size (`float`, defaults to 20):
                Split long audio to multiple segments of `chunk_size` (in seconds) to avoid OOM.
            context_size (`float`, defaults to 2.5):
                Context length (in seconds) to added in each audio chunk for smoother recognition.
            hot_words (List[str], *Optional*):
        """
        super(HuggingFaceASRModel, self).__init__()
        self.model_path = model_path
        self.lm_path = lm_path
        self.vocab_path = vocab_path
        self.hot_words = hot_words
        self.sampling_rate = sampling_rate
        self.token_stride = int(STRIDES * sampling_rate / 1e3)
        self.token_frame_length = int(WINDOW_SIZE * sampling_rate / 1e3)
        self.chunk_size = int(chunk_size * self.sampling_rate)
        self.context_size = int(context_size * self.sampling_rate)

        if self.context_size % self.token_stride != 0:
            self.context_size = self.context_size - (self.context_size % self.token_stride)

        self.context_tokens = self.context_size // self.token_stride

        print("Loading model...")
        start = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)
        self.model = AutoModelForCTC.from_pretrained(model_path, cache_dir=cache_dir).eval().to(self.device)
        print("Model loaded successfully in %fs" % (time.time() - start))

        # Sanity check
        x = torch.zeros([1, 4000]).to(self.device)
        with torch.no_grad():
            out = self.model(x).logits
            self.vocab_size = out.shape[-1]

        self.decoder = self.build_lm(self.processor.tokenizer, self.vocab_path)
        print("Language model loaded successfully in %fs" % (time.time() - start))

    def forward(self, audio):
        if len(audio.shape) == 1:
            audio = [audio]
        elif len(audio.shape) > 2:
            raise ValueError("Expected 2 dimensions input, but got %d dimensions" % len(audio.shape))

        inputs = self.processor(list(audio), sampling_rate=16_000, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            attention_mask = None if not hasattr(inputs, 'attention_mask') else inputs.attention_mask
            logits = self.model(
                inputs.input_values,
                attention_mask=attention_mask,
            ).logits

        return logits

    def forward_streaming(self, audio):
        if len(audio.shape) == 1:
            audio = [audio]
        elif len(audio.shape) > 2:
            raise ValueError("Expected 2 dimensions input, but got %d dimensions" % len(audio.shape))

        inputs = self.processor(list(audio), sampling_rate=16_000, return_tensors="pt", padding=True)
        all_logits = []
        with torch.no_grad():
            audio_chunks = self._split_chunks(inputs.input_values)

            for chunk in audio_chunks:
                all_logits.append(self.model(chunk.to(self.device)).logits.detach().cpu())

            logits = self._merge_chunks(all_logits)

        return logits

    @property
    def inputs_to_logits_ratio(self):
        return functools.reduce(operator.mul, self.model.config.conv_stride, 1)


class SpeechbrainASRModel(BaseASRModel):
    def __init__(
        self,
        model_path,
        lm_path=None,
        vocab_path=None,
        cache_dir=None,
        sampling_rate=16_000,
        chunk_size=20,
        context_size=2.5,
        hot_words=[],
        **kwargs,
    ):
        r"""
        Args:
            model_path (`Union[str, os.PathLike]`):
                Path to pretrained model.
            lm_path (`Union[str, os.PathLike]`, *Optional*):
                Path to language model file.
            vocab_path (`Union[str, os.PathLike]`, *Optional*):
                Path to vocabulary file.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            sampling_rate (`int`, defaults to 16000):
                Sampling rate of audio inputs.
            chunk_size (`float`, defaults to 20):
                Split long audio to multiple segments of `chunk_size` (in seconds) to avoid OOM.
            context_size (`float`, defaults to 2.5):
                Context length (in seconds) to added in each audio chunk for smoother recognition.
            hot_words (List[str], *Optional*):
                Improve domain specificity by adding important contextual words ("hotwords") during inference.
        """
        super(SpeechbrainASRModel, self).__init__()
        self.model_path = model_path
        self.lm_path = lm_path
        self.vocab_path = vocab_path
        self.hot_words = hot_words
        self.sampling_rate = sampling_rate
        self.token_stride = int(STRIDES * sampling_rate / 1e3)
        self.token_frame_length = int(WINDOW_SIZE * sampling_rate / 1e3)
        self.chunk_size = int(chunk_size * self.sampling_rate)
        self.context_size = int(context_size * self.sampling_rate)

        if self.context_size % self.token_stride != 0:
            self.context_size = self.context_size - (self.context_size % self.token_stride)

        self.context_tokens = self.context_size // self.token_stride

        print("Loading model...")
        start = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EncoderASR.from_hparams(source=model_path, savedir=cache_dir, run_opts={'device': self.device})
        print("Model loaded successfully in %fs" % (time.time() - start))

        # Sanity check
        x = torch.zeros([1, 4000]).to(self.device)
        with torch.no_grad():
            out = self.model(x, torch.ones(1))
            self.vocab_size = out.shape[-1]

        self.decoder = self.build_lm(self.model.tokenizer, self.vocab_path)
        print("Language model loaded successfully in %fs" % (time.time() - start))

    def forward(self, audio):
        if len(audio.shape) == 1:
            audio = [audio]
        elif len(audio.shape) > 2:
            raise ValueError("Expected 2 dimensions input, but got %d dimensions" % len(audio.shape))

        inputs = torch.from_numpy(np.asarray(audio))
        input_lens = torch.ones(inputs.shape[0])
        with torch.no_grad():
            logits = self.model(inputs, input_lens)

        return logits

    def forward_streaming(self, audio):
        if len(audio.shape) == 1:
            audio = [audio]
        elif len(audio.shape) > 2:
            raise ValueError("Expected 2 dimensions input, but got %d dimensions" % len(audio.shape))

        audio_inputs = torch.from_numpy(np.asarray(audio))
        all_logits = []
        with torch.no_grad():
            audio_chunks = self._split_chunks(audio_inputs)

            for chunk in audio_chunks:
                inputs = chunk
                input_lens = torch.ones(inputs.shape[0])
                all_logits.append(self.model(inputs, input_lens).detach().cpu())

            logits = self._merge_chunks(all_logits)

        return logits
