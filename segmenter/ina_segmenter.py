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
    instance = None
    def __init__(self, **kwargs):
        super(InaSegmenter, self).__init__(**kwargs)
        self.sampling_rate = 16000
        self.channels = 1

    def __call__(self, media, audio_format=DEFAULT_FORMAT, tmpdir=None, start_sec=None, stop_sec=None):
        if isinstance(media, str):
            wav_file = wave.open(media)
            audio_format = read_audio_format_from_wav_file(wav_file)
            if audio_format.rate != self.sampling_rate:
                raise ValueError(
                    'Ina VAD-splitting only supported for sample rates 16000')
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
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = InaSegmenter(detect_gender=False, batch_size=128)
        
        return cls.instance