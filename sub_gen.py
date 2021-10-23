import os
import subprocess

from audio import AudioFile
from utils import extract_audio, convert_audio, write_to_file
import tqdm

VIDEO_EXT = ['mp4', 'ogg', 'm4v', 'm4a', 'webm', 'flv', 'amv', 'avi', '']
AUDIO_EXT = ['mp3', 'flac', 'wav', 'aac', 'm4a', 'weba', 'sdt']


class SubGenerator:
    def __init__(self, file_path, asr_model, split_duration=5000, max_words=12, sub_format=['srt'], output_directory="./temp"):
        super(SubGenerator, self).__init__()

        if not os.path.exists(file_path):
            raise ValueError("File does not exist: %s" % file_path)

        self.file_path = file_path
        self.file_name, self.file_ext = os.path.split(file_path)[-1].split(".")
        self.temp_path = os.path.join("temp", self.file_name + ".wav")
        self.model = asr_model
        self.split_duration = split_duration
        self.max_words = max_words

        if not os.path.exists("temp"):
            os.makedirs("temp")

        if self.file_ext in VIDEO_EXT:
            self.is_video = True
            extract_audio(self.file_path, self.temp_path)
        elif self.file_ext in AUDIO_EXT:
            self.is_video = False
            convert_audio(self.file_path, self.temp_path)
        else:
            raise ValueError("Extension mismatch")

        self.sub_format = sub_format
        self.output_directory = output_directory
        self.output_file_handle_dict = {}

    def create_sub(self):
        self.audio_file = AudioFile(self.temp_path)
        for format in self.sub_format:
            output_filename = os.path.join(
                self.output_directory, self.file_name + "." + format)
            print("Creating file: " + output_filename)
            self.output_file_handle_dict[format] = open(
                output_filename, mode="w", encoding="utf-8")
            # For VTT format, write header
            if format == "vtt":
                self.output_file_handle_dict[format].write("WEBVTT\n")
                self.output_file_handle_dict[format].write(
                    "Kind: captions\n\n")
        progress_bar = tqdm.tqdm(
            total=int(self.audio_file.audio_length * 1000))
        line_count = 1
        last = 0
        for start, end, audio in self.audio_file.split():
            transcript, tokens, score = self.model.transcribe_with_metadata(audio, start)[
                0]

            if end - start > self.split_duration:
                infer_text = ""
                num_inferred = 0
                prev_end = start

                for token in tokens:
                    infer_text += token['text'] + " "
                    num_inferred += 1
                    if num_inferred > self.max_words or token['end'] - prev_end > self.split_duration:
                        write_to_file(self.output_file_handle_dict, infer_text,
                                      line_count, (prev_end / 1000, token['start'] / 1000))
                        infer_text = ""
                        num_inferred = 0
                        prev_end = token['end']
                        line_count += 1

                if infer_text:
                    write_to_file(self.output_file_handle_dict, infer_text,
                                  line_count, (prev_end / 1000, token['start'] / 1000))
                    line_count += 1
            else:
                write_to_file(self.output_file_handle_dict, transcript,
                              line_count, (start / 1000, end / 1000))
                line_count += 1

            progress_bar.update(int(end - last))
            last = end

        self.audio_file.close()
        self.close_file()
        os.remove(self.temp_path)

    def sync_sub(self):
        if "srt" not in self.sub_format:
            return
        srt_path = os.path.join(self.output_directory, self.file_name + ".srt")
        sync_path = os.path.join(
            self.output_directory, self.file_name + "_synchronized.srt")
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
            srt_path = os.path.join(
                self.output_directory, self.file_name + ".srt")
            out_path = os.path.join(
                self.output_directory, self.file_name + "_sub.mp4")
            cmd = f"ffmpeg -loglevel quiet -i {self.file_path} -i {srt_path} -y -c copy -c:s mov_text {out_path}"
            os.system(cmd)
