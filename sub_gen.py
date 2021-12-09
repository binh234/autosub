import os
import subprocess
import tqdm

from asr.audio import AudioFile
from utils import extract_audio, convert_audio, write_to_file

VIDEO_EXT = ['mp4', 'ogg', 'm4v', 'm4a', 'webm', 'flv', 'amv', 'avi']
AUDIO_EXT = ['mp3', 'flac', 'wav', 'aac', 'm4a', 'weba', 'sdt']


class SubGenerator:
    def __init__(self, 
        file_path, 
        asr_model,
        normalizer,
        gector=None,
        src_lang='vi', 
        split_threshold=200,
        split_duration=5000, 
        max_words=12, 
        sub_format=['srt'], 
        output_directory="./temp"):
        super(SubGenerator, self).__init__()

        if not os.path.exists(file_path):
            raise ValueError("File does not exist: %s" % file_path)

        self.file_path = file_path
        self.file_name, self.file_ext = os.path.split(file_path)[-1].split(".")
        self.max_words = max_words
        self.temp_path = os.path.join("temp", self.file_name + ".wav")
        self.split_duration = split_duration
        self.split_threshold = split_threshold
        self.src_lang = src_lang

        self.model = asr_model
        self.itn = normalizer
        self.gector = gector
        if gector is not None:
            self.max_len = gector.max_len * 2
        else:
            self.max_len = 32

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
    
    def post_process(self, tokens):
        final_tokens = self.itn.inverse_normalize_with_metadata(tokens, verbose=False)
        if self.gector:
            final_batch, _ = self.gector.handle_batch_with_metadata([final_tokens])
            final_tokens = final_batch[0]
        final_transcript = " ".join([token['text'] for token in final_tokens])

        return final_transcript, final_tokens

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
                self.output_file_handle_dict[format].write("Kind: captions\n\n")
        progress_bar = tqdm.tqdm(
            total=int(self.audio_file.audio_length * 1000))
        line_count = 1
        last = 0
        trans_dict = None
        for start, end, audio in self.audio_file.split():
            _, tokens, _ = self.model.transcribe_with_metadata(audio, start)[0]
            if trans_dict is not None:
                if start - trans_dict.get('end', 0) > self.split_threshold or len(trans_dict['tokens']) > self.max_len:
                    final_transcript, final_tokens = self.post_process(trans_dict['tokens'])
                    line_count = self.write_sub(final_transcript, final_tokens, trans_dict['start'], trans_dict['end'], line_count)
                    trans_dict = None
                else:
                    trans_dict['tokens'].extend(tokens)
                    trans_dict['end'] = end
            
            if trans_dict is None:
                trans_dict = {
                    'tokens': tokens,
                    'start': start,
                    'end': end,
                }

            progress_bar.update(int(end - last))
            last = end
        
        if trans_dict is not None:
            final_transcript, final_tokens = self.post_process(trans_dict['tokens'])
            line_count = self.write_sub(final_transcript, final_tokens, start, end, line_count)

        self.audio_file.close()
        self.close_file()

        if os.path.exists(self.temp_path):
            os.remove(self.temp_path)
    
    def write_sub(self, transcript, tokens, start, end, line_count):
        if end - start > self.split_duration:
            infer_text = ""
            num_inferred = 1
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
        
        return line_count

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
