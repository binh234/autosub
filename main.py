from sub_gen import SubGenerator
from asr.model import HuggingFaceASRModel, SpeechbrainASRModel
from gector.gec_model import GecBERTModel
from text_processing.inverse_normalize import InverseNormalizer

file_path = "sample/snews.mp4"
lm_path = None
vocab_path = None
cache_dir = None
segment_backend = "vad"

normalizer = InverseNormalizer("vi")

print(normalizer.inverse_normalize("ngày ba mươi tháng tư năm hai không hai mốt", verbose=True))
print(normalizer.inverse_normalize("năm trăm việt nam đồng", verbose=True))
print(normalizer.inverse_normalize("năm mươi đồng", verbose=True))
print(normalizer.inverse_normalize("bốn năm năm", verbose=True))

# model = HuggingFaceASRModel("pretrain/base", lm_path, vocab_path, cache_dir=cache_dir)
# model = SpeechbrainASRModel("pretrain/base", lm_path, vocab_path, cache_dir=cache_dir)
# gector = GecBERTModel(
#     vocab_path="pretrain/vibert/vocabulary",
#     model_paths=["pretrain/vibert"],
#     device=None,
#     max_len=64,
#     min_len=3,
#     iterations=3,
#     min_error_probability=0.2,
#     lowercase_tokens=False,
#     log=False,
#     confidence=0,
#     weights=None,
#     split_chunk=True,
#     chunk_size=48,
#     overlap_size=16,
#     min_words_cut=8,
# )

# # Recomend_parameters
# # (segment_backend='vad', classify_segment=False)
# # (segment_backend='vad', classify_segment=True, transcribe_music=True)
# # # `Ina` backend may misclassify for speech over noise segments (accuracy around 94%)
# # (segment_backend='ina', classify_segment=True, transcribe_music=True)
# # # a bit downgrade in quality for audios that have many `speech over music` segments like films or gameshows
# # (segment_backend='ina', classify_segment=True, transcribe_music=False)
# gen = SubGenerator(model, normalizer, gector=gector)
# subtitle_paths = gen.create_sub(
#     file_path,
#     sub_format=["srt"],
#     segment_backend="vad",
#     classify_segment=True,
#     show_progress=True,
#     transcribe_music=False,
# )
# print(subtitle_paths)
# srt_path = subtitle_paths[0]
# gen.sync_sub(file_path, srt_path)
# gen.add_sub_to_video(file_path, srt_path)

# gen = SubGenerator("sample/snews.mp4", model, gector, normalizer)
# gen.create_sub()
# gen.sync_sub()
