import os
import subprocess
import tqdm
import tempfile

from asr.audio import AudioFile
from utils import extract_audio, convert_audio, write_to_file, DEFAULT_TEMP_DIR

VIDEO_EXT = ['mp4', 'ogg', 'm4v', 'm4a', 'webm', 'flv', 'amv', 'avi']
AUDIO_EXT = ['mp3', 'flac', 'wav', 'aac', 'm4a', 'weba', 'sdt']


class FileGenObject(object):
    def __init__(self, file_path, sub_format=['srt'], output_directory="./temp", temp_dir=DEFAULT_TEMP_DIR):
        self.file_path = file_path
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        self.temp_dir = temp_dir
        self.temp_path = os.path.join(self.temp_dir, self.file_name + ".wav")

        if self.file_ext in VIDEO_EXT:
            self.is_video = True
            extract_audio(self.file_path, self.temp_path)
        elif self.file_ext in AUDIO_EXT:
            self.is_video = False
            convert_audio(self.file_path, self.temp_path)
        else:
            raise ValueError("Extension mismatch")

        self.sub_format = sub_format
        if output_directory is None:
            output_directory = self.temp_dir
        self.output_directory = output_directory
        self.output_file_handle_dict = {}


class SubGenerator:
    def __init__(
        self,
        file_path,
        asr_model,
        normalizer,
        gector=None,
        src_lang='vi',
        split_threshold_ms=200,
        split_duration_ms=5000,
        min_words=3,
        max_words=12,
        sub_format=['srt'],
        output_directory="./temp",
        segment_backend='vad',
        classify_segment=True,
    ):
        super(SubGenerator, self).__init__()

        if not os.path.exists(file_path):
            raise ValueError("File does not exist: %s" % file_path)

        self.file_path = file_path
        self.file_name, self.file_ext = os.path.split(file_path)[-1].split(".")
        self.min_words = min_words
        self.max_words = max_words
        self.temp_dir = DEFAULT_TEMP_DIR
        self.temp_path = os.path.join(self.temp_dir, self.file_name + ".wav")
        self.allow_tags = {"speech", "male", "female", "noisy_speech", "music"}
        self.split_duration_ms = split_duration_ms
        self.split_threshold_ms = split_threshold_ms
        self.src_lang = src_lang
        self.segment_backend = segment_backend
        self.classify_segment = classify_segment

        self.model = asr_model
        self.itn = normalizer
        self.gector = gector
        if gector is not None:
            self.max_len = gector.max_len * 2
        else:
            self.max_len = 32

        if self.file_ext in VIDEO_EXT:
            self.is_video = True
            extract_audio(self.file_path, self.temp_path)
        elif self.file_ext in AUDIO_EXT:
            self.is_video = False
            convert_audio(self.file_path, self.temp_path)
        else:
            raise ValueError("Extension mismatch")

        self.sub_format = sub_format
        if output_directory is None:
            output_directory = self.temp_dir
        self.output_directory = output_directory
        self.output_file_handle_dict = {}

    def post_process(self, tokens):
        final_tokens = self.itn.inverse_normalize_with_metadata(tokens, verbose=False)
        if self.gector:
            final_batch, _ = self.gector.handle_batch_with_metadata([final_tokens])
            final_tokens = final_batch[0]
        final_transcript = " ".join([token['text'] for token in final_tokens])

        return final_transcript, final_tokens

    def create_sub(self, show_progress=False, transcribe_music=False):
        self.audio_file = AudioFile(self.temp_path)
        for format in self.sub_format:
            output_filename = os.path.join(self.output_directory, self.file_name + "." + format)
            print("Creating file: " + output_filename)
            self.output_file_handle_dict[format] = open(output_filename, mode="w", encoding="utf-8")
            # For VTT format, write header
            if format == "vtt":
                self.output_file_handle_dict[format].write("WEBVTT\n")
                self.output_file_handle_dict[format].write("Kind: captions\n\n")

        if show_progress:
            progress_bar = tqdm.tqdm(total=int(self.audio_file.audio_length * 1000))
        line_count = 1
        last = 0
        trans_dict = None
        for (start, end, audio, tag) in self.audio_file.split(backend=self.segment_backend):
            if tag not in self.allow_tags:
                continue
            if tag == "music" and not transcribe_music:
                if trans_dict is not None:
                    final_transcript, final_tokens = self.post_process(trans_dict['tokens'])
                    line_count = self.write_sub(
                        final_transcript,
                        final_tokens,
                        trans_dict['start'],
                        trans_dict['end'],
                        line_count,
                        trans_dict['split_times'],
                    )
                    trans_dict = None
                write_to_file(self.output_file_handle_dict, "[âm nhạc]", line_count, (start / 1000, end / 1000))
                continue

            _, tokens, _ = self.model.transcribe_with_metadata(audio, start)[0]
            if trans_dict is not None:
                if (
                    len(tokens) == 0
                    or start - trans_dict.get('end', 0) > self.split_threshold_ms
                    or len(trans_dict['tokens']) > self.max_len
                ):
                    final_transcript, final_tokens = self.post_process(trans_dict['tokens'])
                    line_count = self.write_sub(
                        final_transcript,
                        final_tokens,
                        trans_dict['start'],
                        trans_dict['end'],
                        line_count,
                        trans_dict['split_times'],
                    )
                    trans_dict = None
                else:
                    trans_dict['tokens'].extend(tokens)
                    trans_dict['split_times'].append(trans_dict['end'])
                    trans_dict['end'] = end

            if trans_dict is None:
                trans_dict = {
                    'tokens': tokens,
                    'start': start,
                    'end': end,
                    'split_times': [],
                }

            if show_progress:
                progress_bar.update(int(end - last))
            last = end

        if trans_dict is not None:
            final_transcript, final_tokens = self.post_process(trans_dict['tokens'])
            line_count = self.write_sub(
                final_transcript,
                final_tokens,
                trans_dict['start'],
                trans_dict['end'],
                line_count,
                trans_dict['split_times'],
            )
        if show_progress:
            progress_bar.update(int(progress_bar.total - last))
        self.audio_file.close()
        self.close_file()

        if os.path.exists(self.temp_path):
            os.remove(self.temp_path)

    def write_sub(self, transcript, tokens, start, end, line_count, split_times=[]):
        if split_times is None:
            split_times = []
        split_times.append(1e8)
        num_tokens = len(tokens)

        if num_tokens == 0:
            return line_count

        split_tokens = [[] for _ in range(len(split_times))]
        split_idx = 0

        for token in tokens:
            while token['start'] > split_times[split_idx]:
                split_idx += 1
            split_tokens[split_idx].append(token)

        for sub_tokens in split_tokens:
            num_sub_tokens = len(sub_tokens)
            if num_sub_tokens == 0:
                continue
            start = sub_tokens[0]["start"]
            end = sub_tokens[-1]["end"]

            if end - start > self.split_duration_ms:
                last_tokens_length = num_sub_tokens % self.max_words
                num_lines = num_sub_tokens // self.max_words
                if last_tokens_length >= self.min_words or num_lines == 0:
                    num_lines += 1

                for i in range(num_lines):
                    if i == num_lines - 1:
                        token_batch = sub_tokens[i * self.max_words :]
                    else:
                        token_batch = sub_tokens[i * self.max_words : (i + 1) * self.max_words]
                    infer_text = " ".join([token["text"] for token in token_batch])
                    write_to_file(
                        self.output_file_handle_dict,
                        infer_text,
                        line_count,
                        (token_batch[0]["start"] / 1000, token_batch[-1]["end"] / 1000),
                    )
                    line_count += 1
            else:
                text = " ".join([token["text"] for token in sub_tokens])
                write_to_file(self.output_file_handle_dict, text, line_count, (start / 1000, end / 1000))
                line_count += 1

        return line_count

    def sync_sub(self):
        if "srt" not in self.sub_format:
            return
        srt_path = os.path.join(self.output_directory, self.file_name + ".srt")
        sync_path = os.path.join(self.output_directory, self.file_name + "_synchronized.srt")
        cmd = f"ffsubsync {self.file_path} -i {srt_path} -o {sync_path}"

        subprocess.call(cmd, shell=True)

        if os.path.exists(sync_path):
            os.remove(srt_path)
            os.rename(sync_path, srt_path)

    def close_file(self):
        for format in self.output_file_handle_dict.keys():
            self.output_file_handle_dict[format].close()

    def add_sub_to_video(self):
        if self.is_video:
            srt_path = os.path.join(self.output_directory, self.file_name + ".srt")
            out_path = os.path.join(self.output_directory, self.file_name + "_sub.mp4")
            cmd = f"ffmpeg -loglevel quiet -i {self.file_path} -i {srt_path} -y -c copy -c:s mov_text {out_path}"
            os.system(cmd)
