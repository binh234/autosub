import collections
import datetime
import numpy as np
import os
import subprocess
import tempfile
from collections import namedtuple

AudioFormat = namedtuple('AudioFormat', 'rate channels width')

DEFAULT_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_WIDTH = 2
DEFAULT_FORMAT = AudioFormat(DEFAULT_RATE, DEFAULT_CHANNELS, DEFAULT_WIDTH)
DEFAULT_TEMP_DIR = tempfile.mkdtemp()


def extract_audio(video_path, dst_path, format=DEFAULT_FORMAT):
    """
    Extracts audio from video file
    """
    cmd = f'ffmpeg -loglevel quiet -i "{video_path}" -y -hide_banner -ac {format.channels} -ar {format.rate} -vn {dst_path} '
    # print(cmd)
    subprocess.call(cmd, shell=True)

    return dst_path


def convert_audio(audio_path, dst_path, format=DEFAULT_FORMAT):
    """
    Convert audio file to format
    """
    cmd = f'ffmpeg -loglevel quiet -i "{audio_path}" -y -hide_banner -ac {format.channels} -ar {format.rate} {dst_path}'
    # print(cmd)
    subprocess.call(cmd, shell=True)

    return dst_path


def read_audio_format_from_wav_file(wav_file):
    return AudioFormat(wav_file.getframerate(), wav_file.getnchannels(), wav_file.getsampwidth())


def get_num_samples(pcm_buffer_size, audio_format=DEFAULT_FORMAT):
    return pcm_buffer_size // (audio_format.channels * audio_format.width)


def get_pcm_duration(pcm_buffer_size, audio_format=DEFAULT_FORMAT):
    """Calculates duration in seconds of a binary PCM buffer (typically read from a WAV file)"""
    return get_num_samples(pcm_buffer_size, audio_format) / audio_format.rate


def get_dtype(audio_format):
    if audio_format.width not in [1, 2, 4]:
        raise ValueError('Unsupported sample width: {}'.format(audio_format.width))
    return [None, np.int8, np.int16, None, np.int32][audio_format.width]


def pcm_to_np(pcm_data, audio_format=DEFAULT_FORMAT):
    """
    Converts PCM data (e.g. read from a wavfile) into a mono numpy column vector
    with values in the range [0.0, 1.0].
    """
    # Handles both mono and stereo audio
    dtype = get_dtype(audio_format)
    samples = np.frombuffer(pcm_data, dtype=dtype)

    # Read interleaved channels
    nchannels = audio_format.channels
    samples = samples.reshape((int(len(samples) / nchannels), nchannels))

    # Convert to 0.0-1.0 range
    samples = samples.astype(np.float32) / np.iinfo(dtype).max

    # Average multi-channel clips into mono and turn into column vector
    return np.expand_dims(np.mean(samples, axis=1), axis=1)


def read_frames(wav_file, frame_duration_ms=30, yield_remainder=False):
    audio_format = read_audio_format_from_wav_file(wav_file)
    frame_size = int(audio_format.rate * (frame_duration_ms / 1000.0))
    while True:
        try:
            data = wav_file.readframes(frame_size)
            if not yield_remainder and get_pcm_duration(len(data), audio_format) * 1000 < frame_duration_ms:
                break
            yield data
        except EOFError:
            break


def read_frames_from_file(wav_file, audio_format=DEFAULT_FORMAT, frame_duration_ms=30, yield_remainder=False):
    for frame in read_frames(wav_file, frame_duration_ms=frame_duration_ms, yield_remainder=yield_remainder):
        yield frame


def vad_split(audio_frames, audio_format=DEFAULT_FORMAT, num_padding_frames=10, threshold=0.5, aggressiveness=3):
    from webrtcvad import Vad  # pylint: disable=import-outside-toplevel

    if audio_format.channels != 1:
        raise ValueError('VAD-splitting requires mono samples')
    if audio_format.width != 2:
        raise ValueError('VAD-splitting requires 16 bit samples')
    if audio_format.rate not in [8000, 16000, 32000, 48000]:
        raise ValueError('VAD-splitting only supported for sample rates 8000, 16000, 32000, or 48000')
    if aggressiveness not in [0, 1, 2, 3]:
        raise ValueError('VAD-splitting aggressiveness mode has to be one of 0, 1, 2, or 3')
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    vad = Vad(int(aggressiveness))
    voiced_frames = []
    frame_duration_ms = 0
    frame_index = 0
    for frame_index, frame in enumerate(audio_frames):
        frame_duration_ms = get_pcm_duration(len(frame), audio_format) * 1000
        if int(frame_duration_ms) not in [10, 20, 30]:
            raise ValueError('VAD-splitting only supported for frame durations 10, 20, or 30 ms')
        is_speech = vad.is_speech(frame, audio_format.rate)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > threshold * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > threshold * ring_buffer.maxlen:
                triggered = False
                yield b''.join(voiced_frames), frame_duration_ms * max(
                    0, frame_index - len(voiced_frames)
                ), frame_duration_ms * frame_index
                ring_buffer.clear()
                voiced_frames = []
    if len(voiced_frames) > 0:
        yield b''.join(voiced_frames), frame_duration_ms * max(
            0, frame_index - len(voiced_frames)
        ), frame_duration_ms * (frame_index + 1)


def split_audio_file(
    audio_path,
    audio_format=DEFAULT_FORMAT,
    aggressiveness=3,
):
    def generate_values():
        frames = read_frames_from_file(audio_path)
        segments = vad_split(frames, aggressiveness=aggressiveness)
        for segment in segments:
            segment_buffer, time_start, time_end = segment
            samples = pcm_to_np(segment_buffer, audio_format)
            yield time_start, time_end, np.squeeze(samples)

    return generate_values()


def get_timestamp_string(timedelta, format):
    """Convert the timedelta into something that can be used by a subtitle file.

    Args:
        timedelta : timedelta timestmap
        format : subtitle format
    """
    sep = '.' if format == "vtt" else ','
    # timedelta may be eg, '0:00:14'
    if '.' in str(timedelta):
        timestamp = "0" + str(timedelta).split(".")[0] + sep + str(timedelta).split(".")[-1][:3]
    else:
        timestamp = "0" + str(timedelta) + sep + "000"
    return timestamp


def write_to_file(output_file_handle_dict, inferred_text, line_count, limits):
    """Write the inferred text to SRT file
    Follows a specific format for SRT files

    Args:
        output_file_handle_dict : Mapping of subtitle format (eg, 'srt') to open file_handle
        inferred_text : text to be written
        line_count : subtitle line count
        limits : starting and ending times for text
    """

    for format in output_file_handle_dict.keys():
        from_dur = get_timestamp_string(datetime.timedelta(seconds=float(limits[0])), format)
        to_dur = get_timestamp_string(datetime.timedelta(seconds=float(limits[1])), format)

        file_handle = output_file_handle_dict[format]
        if format == 'srt':
            file_handle.write(str(line_count) + "\n")
            file_handle.write(from_dur + " --> " + to_dur + "\n")
            file_handle.write(inferred_text + "\n\n")
        elif format == 'vtt':
            file_handle.write(from_dur + " --> " + to_dur + " align:start position:0%\n")
            file_handle.write(inferred_text + "\n\n")
        elif format == 'txt':
            file_handle.write(inferred_text + "\n")
