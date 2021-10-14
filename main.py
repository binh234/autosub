import os
import argparse

from model import ASRModel
from sub_gen import SubGenerator

file_path = "sample/anime.mp4"

model = ASRModel("./pretrain/base", "./pretrain/lm.bin")

gen = SubGenerator(file_path, model)
gen.create_sub()
gen.sync_sub()
# gen.add_sub_to_video()

gen = SubGenerator("sample/snews.mp4", model)
gen.create_sub()
gen.sync_sub()
