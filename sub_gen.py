import os
import subprocess
import uuid
import warnings
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
        asr_model,
        normalizer,
        gector=None,
        src_lang='vi',
    ):
        super(SubGenerator, self).__init__()
        self.temp_dir = DEFAULT_TEMP_DIR
        self.allow_tags = {"speech", "male", "female", "noisy_speech", "music"}
        self.src_lang = src_lang
        self.model = asr_model
        self.itn = normalizer
        self.gector = gector
        if gector is not None:
            self.max_len = gector.max_len * 2
        else:
            self.max_len = 32

    def post_process(self, tokens):
        final_tokens = self.itn.inverse_normalize_with_metadata(tokens, verbose=False)
        if self.gector:
            final_batch, _ = self.gector.handle_batch_with_metadata([final_tokens])
            final_tokens = final_batch[0]

        return final_tokens

    def create_sub(
        self,
        file_path,
        split_threshold_ms=200,
        split_duration_ms=5000,
        min_words=3,
        max_words=12,
        sub_format=['srt'],
        output_directory="./temp",
        segment_backend='vad',
        classify_segment=False,
        show_progress=False,
        transcribe_music=False,
    ):

        if not os.path.exists(file_path):
            raise ValueError("File does not exist: %s" % file_path)

        if classify_segment and segment_backend == 'vad':
            warnings.warn(
                "Classify segment should be used with ina backend, otherwise the transcript quality might be downgraded"
            )

        file_name, file_ext = os.path.split(file_path)[-1].split(".")
        file_id = uuid.uuid4().hex
        temp_path = os.path.join(self.temp_dir, file_id + ".wav")
        output_file_handle_dict = {}
        output_files = []

        if file_ext in VIDEO_EXT:
            extract_audio(file_path, temp_path)
        elif file_ext in AUDIO_EXT:
            convert_audio(file_path, temp_path)
        else:
            raise ValueError(f"Extension mismatch, should be one of {AUDIO_EXT + VIDEO_EXT}")

        if output_directory is None:
            warnings.warn(f"Output directory is None, using {self.temp_dir} instead.")
            output_directory = self.temp_dir

        audio_file = AudioFile(temp_path)
        for format in sub_format:
            output_filename = os.path.join(output_directory, file_name + "." + format)
            output_files.append(output_filename)
            print("Creating file: " + output_filename)
            output_file_handle_dict[format] = open(output_filename, mode="w", encoding="utf-8")
            # For VTT format, write header
            if format == "vtt":
                output_file_handle_dict[format].write("WEBVTT\n")
                output_file_handle_dict[format].write("Kind: captions\n\n")

        if show_progress:
            progress_bar = tqdm.tqdm(total=int(audio_file.audio_length * 1000))

        line_count = 1
        last = 0
        trans_dict = None
        subtitle_kwargs = {
            "output_file_handle_dict": output_file_handle_dict,
            "min_words": min_words,
            "max_words": max_words,
            "split_duration_ms": split_duration_ms,
        }
        for (start, end, audio, tag) in audio_file.split(backend=segment_backend, classify=classify_segment):
            if tag not in self.allow_tags:
                continue
            if tag == "music" and not transcribe_music:
                if trans_dict is not None:
                    final_tokens = self.post_process(trans_dict['tokens'])
                    line_count = self.write_sub(
                        final_tokens,
                        trans_dict['start'],
                        trans_dict['end'],
                        line_count,
                        trans_dict['split_times'],
                        **subtitle_kwargs,
                    )
                    trans_dict = None
                write_to_file(output_file_handle_dict, "[âm nhạc]", line_count, (start / 1000, end / 1000))
                line_count += 1
                continue

            _, tokens, _ = self.model.transcribe_with_metadata(audio, start)[0]
            if trans_dict is not None:
                if (
                    len(tokens) == 0
                    or start - trans_dict.get('end', 0) > split_threshold_ms
                    or len(trans_dict['tokens']) > self.max_len
                ):
                    final_tokens = self.post_process(trans_dict['tokens'])
                    line_count = self.write_sub(
                        final_tokens,
                        trans_dict['start'],
                        trans_dict['end'],
                        line_count,
                        trans_dict['split_times'],
                        **subtitle_kwargs,
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

        # Handle last batch
        if trans_dict is not None:
            final_tokens = self.post_process(trans_dict['tokens'])
            line_count = self.write_sub(
                final_tokens,
                trans_dict['start'],
                trans_dict['end'],
                line_count,
                trans_dict['split_times'],
                **subtitle_kwargs,
            )
        if show_progress:
            progress_bar.update(int(progress_bar.total - last))
        audio_file.close()

        for format in output_file_handle_dict.keys():
            output_file_handle_dict[format].close()

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return output_files

    def write_sub(
        self,
        tokens,
        start,
        end,
        line_count,
        split_times=[],
        output_file_handle_dict={},
        min_words=3,
        max_words=12,
        split_duration_ms=5000,
    ):
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

            if end - start > split_duration_ms:
                last_tokens_length = num_sub_tokens % max_words
                num_lines = num_sub_tokens // max_words
                if last_tokens_length >= min_words or num_lines == 0:
                    num_lines += 1

                for i in range(num_lines):
                    if i == num_lines - 1:
                        token_batch = sub_tokens[i * max_words :]
                    else:
                        token_batch = sub_tokens[i * max_words : (i + 1) * max_words]
                    infer_text = " ".join([token["text"] for token in token_batch])
                    write_to_file(
                        output_file_handle_dict,
                        infer_text,
                        line_count,
                        (token_batch[0]["start"] / 1000, token_batch[-1]["end"] / 1000),
                    )
                    line_count += 1
            else:
                text = " ".join([token["text"] for token in sub_tokens])
                write_to_file(output_file_handle_dict, text, line_count, (start / 1000, end / 1000))
                line_count += 1

        return line_count

    def sync_sub(self, file_path, subtitle_path, output_path=None):
        _, sub_ext = os.path.splitext(subtitle_path)
        if sub_ext != ".srt":
            warnings.warn("Subtitle synchronization only supported for srt format")
            return
        if output_path is None or output_path == subtitle_path:
            output_path = subtitle_path[:-4] + "_synchronized.srt"
            overwrite = True

        cmd = f"ffsubsync {file_path} -i {subtitle_path} -o {output_path}"

        subprocess.call(cmd, shell=True)

        if overwrite:
            os.remove(subtitle_path)
            os.rename(output_path, subtitle_path)

    def add_sub_to_video(self, file_path, subtitle_path, output_path=None):
        file_name, file_ext = os.path.splitext(file_path)
        if output_path is None or output_path == subtitle_path:
            output_path = file_name + "_sub" + file_ext
            overwrite = True
        if file_ext not in VIDEO_EXT:
            raise ValueError(f"Only supported for videoof format {VIDEO_EXT}")
        cmd = f"ffmpeg -loglevel quiet -i {file_path} -i {subtitle_path} -y -c copy -c:s mov_text {output_path}"
        os.system(cmd)

        if overwrite:
            os.remove(file_path)
            os.rename(output_path, file_path)
