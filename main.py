from sub_gen import SubGenerator
from asr.model import ASRModel
from gector.gec_model import GecBERTModel
from text_processing.inverse_normalize import InverseNormalizer

file_path = "sample/audio_num.mp3"

normalizer = InverseNormalizer("vi")

# print(normalizer.inverse_normalize("năm hai không hai mốt", verbose=True))

model = ASRModel("pretrain/base", None)
gector = GecBERTModel(vocab_path="pretrain/phobert/vocabulary",
                      model_paths=["pretrain/phobert/model.th"],
                      device="cpu",
                      max_len=64, min_len=3,
                      iterations=3,
                      min_error_probability=0.2,
                      lowercase_tokens=False,
                      model_name='vinai/phobert-base',
                      special_tokens_fix=1,
                      log=False,
                      confidence=0,
                      is_ensemble=False,
                      weights=None,
                      split_chunk=True,
                      chunk_size=48,
                      overlap_size=8,
                      min_words_cut=4)

gen = SubGenerator(file_path, model, normalizer, gector=gector)
gen.create_sub()
# gen.sync_sub()
# gen.add_sub_to_video()

# gen = SubGenerator("sample/snews.mp4", model, gector, normalizer)
# gen.create_sub()
# gen.sync_sub()
