import numpy as np
import torch
import wave
from audify.modeling_audify import AudifyModel
from segmenter.ina_segmenter import InaSegmenter
from utils import (
    DEFAULT_FORMAT,
    pcm_to_np,
    read_audio_format_from_wav_file,
    read_frames_from_file,
    vad_split,
)


class AudioFile:
    def __init__(self, audio_path, audio_format=DEFAULT_FORMAT, min_segment_len_ms=300):
        super(AudioFile, self).__init__()

        self.audio_path = audio_path

        self.wav_file = wave.open(audio_path)
        self.audio_length = self.wav_file.getnframes() / self.wav_file.getframerate()
        self.audio_format = read_audio_format_from_wav_file(self.wav_file)
        self.min_segment_ms = min_segment_len_ms

        assert self.audio_format == audio_format, "Audio format mismatch"

        self.audify_model = None

    def _ina_split(self, classify=True):
        self.rewind()
        media_buffer = self.wav_file.readframes(self.wav_file.getnframes())
        media = pcm_to_np(media_buffer, self.audio_format).squeeze()
        segmenter = InaSegmenter.get_instance()
        segmentation = segmenter(media)

        if classify == True and self.audify_model is None:
            self.audify_model = AudifyModel.get_instance()

        for (label, time_start, time_end) in segmentation:
            tag = label
            if label in {"speech", "male", "female", "music"} and time_end - time_start > self.min_segment_ms / 1000:
                start_idx = int(time_start * self.audio_format.rate)
                end_idx = int(time_end * self.audio_format.rate)
                samples = media[start_idx:end_idx]

                if label == "music" and classify == True:
                    with torch.no_grad():
                        _, _, _, predict_tags = self.audify_model(
                            torch.FloatTensor(samples).unsqueeze(0), torch.ones(1)
                        )
                        tag = predict_tags[0]

                yield int(time_start * 1000), int(time_end * 1000), samples, tag

    def _vad_split(self, aggressiveness=3, classify=True):
        if classify == True and self.audify_model is None:
            self.audify_model = AudifyModel.get_instance()

        frames = read_frames_from_file(self.wav_file)
        segments = vad_split(frames, aggressiveness=aggressiveness)
        for segment in segments:
            tag = 'speech'
            segment_buffer, time_start, time_end = segment
            if time_end - time_start < self.min_segment_ms:
                continue
            samples = pcm_to_np(segment_buffer, self.audio_format)
            samples = samples.squeeze()

            if classify == True:
                with torch.no_grad():
                    _, _, _, predict_tags = self.audify_model(torch.FloatTensor(samples).unsqueeze(0), torch.ones(1))
                    tag = predict_tags[0]

            yield time_start, time_end, samples, tag

    def split(self, aggressiveness=3, classify=True, backend='vad'):
        if backend == "vad":
            return self._vad_split(aggressiveness, classify)
        elif backend == "ina":
            return self._ina_split(classify)
        else:
            raise ValueError("Please choose `vad` or `ina` backend for segmentation")

    def close(self):
        self.wav_file.close()

    def rewind(self):
        self.wav_file.rewind()
