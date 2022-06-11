# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import Levenshtein

from argparse import ArgumentParser
from time import perf_counter
from typing import List
from difflib import SequenceMatcher

from text_processing.normalize import Normalizer
from text_processing.token_parser import TokenParser


class InverseNormalizer(Normalizer):
    """
    Inverse normalizer that converts text from spoken to written form. Useful for ASR postprocessing.
    Input is expected to have no punctuation outside of approstrophe (') and dash (-) and be lower cased.

    Args:
        lang: language specifying the ITN
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
    """

    def __init__(self, lang: str = 'en', cache_dir: str = None, overwrite_cache: bool = False):
        if lang == 'vi':
            from text_processing.vi.taggers.tokenize_and_classify import ClassifyFst
            from text_processing.vi.verbalizers.verbalize_final import VerbalizeFinalFst
        else:
            raise NotImplementedError

        self.tagger = ClassifyFst(cache_dir=cache_dir, overwrite_cache=overwrite_cache)
        self.verbalizer = VerbalizeFinalFst()
        self.parser = TokenParser()

    def inverse_normalize_list(self, texts: List[str], verbose=False) -> List[str]:
        """
        NeMo inverse text normalizer

        Args:
            texts: list of input strings
            verbose: whether to print intermediate meta information

        Returns converted list of input strings
        """
        return self.normalize_list(texts=texts, verbose=verbose)

    def inverse_normalize(self, text: str, verbose: bool = False) -> str:
        """
        Main function. Inverse normalizes tokens from spoken to written form
            e.g. twelve kilograms -> 12 kg

        Args:
            text: string that may include semiotic classes
            verbose: whether to print intermediate meta information

        Returns: written form
        """
        chunk_size = 512
        tokens = text.split()
        if len(tokens) <= chunk_size:
            return self.normalize(text=text, verbose=verbose)
        else:
            result = ""
            for i in range(0, len(tokens), chunk_size):
                sub_text = " ".join(tokens[i, i + chunk_size])
                result += self.normalize(text=sub_text, verbose=verbose) + " "
            return result.strip()

    def inverse_normalize_list_with_metadata(self, text_metas: List, verbose=False) -> List[str]:
        """
        NeMo inverse text normalizer

        Args:
            texts: list of input strings
            verbose: whether to print intermediate meta information

        Returns converted list of input strings
        """
        res = []
        for input in text_metas:
            try:
                text = self.inverse_normalize_with_metadata(input, verbose=verbose)
            except:
                print(input)
                raise Exception
            res.append(text)
        return res

    def inverse_normalize_with_metadata(self, text_meta: List, verbose: bool = False):
        """
        Main function. Inverse normalizes tokens from spoken to written form
            e.g. twelve kilograms -> 12 kg

        Args:
            text_meta: list of tokens include text, start time, end time and score for each token
            verbose: whether to print intermediate meta information

        Returns: written form
        """
        text = " ".join([token['text'] for token in text_meta])
        normalize_text = self.inverse_normalize(text, verbose=verbose)

        # If no changes are made, return original
        if text == normalize_text:
            return text_meta

        normalize_text_meta = []
        source_tokens = text.split()
        target_tokens = normalize_text.split()
        matcher = SequenceMatcher(None, source_tokens, target_tokens)
        diffs = list(matcher.get_opcodes())

        for diff in diffs:
            tag, i1, i2, j1, j2 = diff

            if tag == "equal":
                normalize_text_meta.extend(text_meta[i1:i2])
            else:
                start = text_meta[i1]['start']
                end = text_meta[i2 - 1]['end']
                num_target_tokens = j2 - j1
                time_step = (end - start) / num_target_tokens
                for c in range(num_target_tokens):
                    normalize_text_meta.append(
                        {
                            'text': target_tokens[j1 + c],
                            'start': start,
                            'end': start + time_step,
                        }
                    )
                    start += time_step

        return normalize_text_meta


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_string", help="input string", type=str)
    parser.add_argument("--language", help="language", choices=['vi'], default="en", type=str)
    parser.add_argument("--verbose", help="print info for debugging", action='store_true')
    parser.add_argument("--overwrite_cache", help="set to True to re-create .far grammar files", action="store_true")
    parser.add_argument(
        "--cache_dir",
        help="path to a dir with .far grammar file. Set to None to avoid using cache",
        default=None,
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_time = perf_counter()
    inverse_normalizer = InverseNormalizer(
        lang=args.language, cache_dir=args.cache_dir, overwrite_cache=args.overwrite_cache
    )
    print(f'Time to generate graph: {round(perf_counter() - start_time, 2)} sec')
    start_time = perf_counter()
    print(inverse_normalizer.inverse_normalize(args.input_string, verbose=args.verbose))
    print(f'Execution time: {round(perf_counter() - start_time, 2)} sec')
