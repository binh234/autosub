import os
import argparse

from asr.model import ASRModel
from sub_gen import SubGenerator

file_path = "sample/audio_wind.mp3"

model = ASRModel("pretrain/base", None)

gen = SubGenerator(file_path, model)
gen.create_sub()
# gen.sync_sub()
# gen.add_sub_to_video()

# gen = SubGenerator("sample/snews.mp4", model)
# gen.create_sub()
# gen.sync_sub()
