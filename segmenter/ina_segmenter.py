from typing import Dict
import numpy as np
from inaSpeechSegmenter import Segmenter
from inaSpeechSegmenter.sidekit_mfcc import mfcc
from utils.utils import DEFAULT_FORMAT, pcm_to_np, read_audio_format_from_wav_file
import wave


def media2feats(waveform):
    """
    Extract features for wav 16k mono
    waveform: NDArray
    """
    # ignore warnings resulting from empty signals parts
    _, loge, _, mspec = mfcc(waveform.astype(np.float32), get_mspec=True)

    # Management of short duration segments
    difflen = 0
    if len(loge) < 68:
        difflen = 68 - len(loge)
        mspec = np.concatenate((mspec, np.ones((difflen, 24)) * np.min(mspec)))

    return mspec, loge, difflen


class InaSegmenter(Segmenter):
    _cache: Dict[str, Segmenter] = {}

    def __init__(self, **kwargs):
        super(InaSegmenter, self).__init__(**kwargs)
        self.sampling_rate = 16000
        self.channels = 1

    def __call__(self, media, audio_format=DEFAULT_FORMAT, tmpdir=None, start_sec=None, stop_sec=None):
        if isinstance(media, str):
            wav_file = wave.open(media)
            audio_format = read_audio_format_from_wav_file(wav_file)
            if audio_format.rate != self.sampling_rate:
                raise ValueError('Ina VAD-splitting only supported for sample rates 16000')
            elif audio_format.channels != self.channels:
                raise ValueError('Ina VAD-splitting requires mono samples')

            media_buffer = wav_file.readframes(wav_file.getnframes())
            media = pcm_to_np(media_buffer, audio_format).squeeze()

        media = np.asarray(media)
        mspec, loge, difflen = media2feats(media)
        if start_sec is None:
            start_sec = 0
        # do segmentation
        return self.segment_feats(mspec, loge, difflen, start_sec)

    @classmethod
    def load(cls, batch_size=128, detect_gender=False, cache_model=True):
        """
        In some instances you may want to load the same model twice
        This factory provides a cache so that you don't actually have to load the model twice.
        """
        key = f'{batch_size}_{detect_gender}'
        if key in cls._cache:
            return cls._cache[key]
        
        model = InaSegmenter(batch_size=batch_size, detect_gender=detect_gender)
        if cache_model:
            cls._cache[key] = model

        return model
