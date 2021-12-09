import os
import argparse

from sub_gen import SubGenerator
from asr.model import ASRModel
from text_processing.inverse_normalize import InverseNormalizer

file_path = "sample/audio_num.mp3"

normalizer = InverseNormalizer("vi")

model = ASRModel("pretrain/base", None)

gen = SubGenerator(file_path, model, normalizer)
gen.create_sub()
# gen.sync_sub()
# gen.add_sub_to_video()

# gen = SubGenerator("sample/snews.mp4", model)
# gen.create_sub()
# gen.sync_sub()
