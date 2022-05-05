"""Wrapper of AllenNLP model. Fixes errors based on model predictions"""
from difflib import SequenceMatcher
import logging
import os
import sys
import re
from time import time
import warnings

import torch
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import util

from gector.bert_token_embedder import PretrainedBertEmbedder
from gector.seq2labels_model import Seq2Labels
from gector.tokenizer_indexer import PretrainedBertIndexer
from gector.wordpiece_indexer import PretrainedBertIndexer as WordpieceIndexer
from utils.helpers import PAD, UNK, get_target_sent_by_edits, START_TOKEN
from utils.helpers import get_weights_name

logging.getLogger("werkzeug").setLevel(logging.ERROR)
logger = logging.getLogger(__file__)


class GecBERTModel(object):
    def __init__(
        self,
        vocab_path=None,
        model_paths=None,
        weights=None,
        device=None,
        max_len=64,
        min_len=3,
        lowercase_tokens=False,
        log=False,
        iterations=3,
        model_name='roberta',
        special_tokens_fix=1,
        is_ensemble=False,
        min_error_probability=0.0,
        confidence=0,
        resolve_cycles=False,
        split_chunk=False,
        chunk_size=48,
        overlap_size=12,
        min_words_cut=6,
        punc_dict={':', ".", ",", "?"},
    ):
        r"""
        Args:
            vocab_path (`str`):
                Path to vocabulary directory.
            model_paths (`List[str]`):
                List of model paths.
            weights (`int`, *Optional*, defaults to None):
                Weights of each model. Only relevant if `is_ensemble is True`.
            device (`int`, *Optional*, defaults to None):
                Device to load model. If not set, device will be automatically choose.
            max_len (`int`, defaults to 64):
                Max sentence length to be processed (all longer will be truncated).
            min_len (`int`, defaults to 3):
                Min sentence length to be processed (all shorted will be returned w/o changes).
            lowercase_tokens (`bool`, defaults to False):
                Whether to lowercase tokens.
            log (`bool`, defaults to False):
                Whether to enable logging.
            iterations (`int`, defaults to 3):
                Max iterations to run during inference.
            special_tokens_fix (`bool`, defaults to True):
               Whether to fix problem with [CLS], [SEP] tokens tokenization.
            is_ensemble (`bool`, defaults to False):
                Whether to do ensembling. Model must be stored in format {model_name}_{special_tokens_fix}.
                Ex: `roberta_1.th`
            min_error_probability (`float`, defaults to `0.0`):
                Minimum probability for each action to apply.
            confidence (`float`, defaults to `0.0`):
                How many probability to add to $KEEP token.
            split_chunk (`bool`, defaults to False):
                Whether to split long sentences to multiple segments of `chunk_size`.
                !Warning: if `chunk_size > max_len`, each segment will be truncate to `max_len`.
            chunk_size (`int`, defaults to 48):
                Length of each segment (in words). Only relevant if `split_chunk is True`.
            overlap_size (`int`, defaults to 12):
                Overlap size (in words) between two consecutive segments. Only relevant if `split_chunk is True`.
            min_words_cut (`int`, defaults to 6):
                Minimun number of words to be cut while merging two consecutive segments.
                Only relevant if `split_chunk is True`.
            punc_dict (List[str], defaults to `{':', ".", ",", "?"}`):
                List of punctuations.
        """
        self.model_weights = list(map(float, weights)) if weights else [1] * len(model_paths)
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        )
        self.max_len = max_len
        self.min_len = min_len
        self.lowercase_tokens = lowercase_tokens
        self.min_error_probability = min_error_probability
        self.vocab = Vocabulary.from_files(vocab_path)
        self.log = log
        self.iterations = iterations
        self.confidence = confidence
        self.resolve_cycles = resolve_cycles

        assert (
            chunk_size > 0 and chunk_size // 2 >= overlap_size
        ), "Chunk merging required overlap size must be smaller than half of chunk size"
        self.split_chunk = split_chunk
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_words_cut = min_words_cut
        self.stride = chunk_size - overlap_size
        self.punc_dict = punc_dict
        # set training parameters and operations

        self.indexers = []
        self.models = []
        for model_path in model_paths:
            if is_ensemble:
                model_name, special_tokens_fix = self._get_model_data(model_path)
            weights_name = get_weights_name(model_name, lowercase_tokens)
            self.indexers.append(self._get_indexer(weights_name, special_tokens_fix))
            model = Seq2Labels(
                vocab=self.vocab,
                text_field_embedder=self._get_embbeder(weights_name, special_tokens_fix),
                confidence=self.confidence,
            )
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model = model.to(self.device)
            model.eval()
            self.models.append(model)

    @staticmethod
    def _get_model_data(model_path):
        model_name = model_path.split('/')[-1]
        tr_model, stf = model_name.split('_')[:2]
        return tr_model, int(stf)

    def _restore_model(self, input_path):
        if os.path.isdir(input_path):
            print("Model could not be restored from directory", file=sys.stderr)
            filenames = []
        else:
            filenames = [input_path]
        for model_path in filenames:
            try:
                loaded_model = torch.load(model_path)
                # else:
                #     loaded_model = torch.load(model_path,
                #                               map_location=lambda storage,
                #                                                   loc: storage)
            except:
                print(f"{model_path} is not valid model", file=sys.stderr)
            own_state = self.model.state_dict()
            for name, weights in loaded_model.items():
                if name not in own_state:
                    continue
                try:
                    if len(filenames) == 1:
                        own_state[name].copy_(weights)
                    else:
                        own_state[name] += weights
                except RuntimeError:
                    continue
        print("Model is restored", file=sys.stderr)

    def split_chunks(self, batch):
        # return batch pairs of indices
        result = []
        indices = []
        for tokens in batch:
            start = len(result)
            num_token = len(tokens)
            if num_token <= self.chunk_size:
                result.append(tokens)
            elif num_token > self.chunk_size and num_token < (self.chunk_size * 2 - self.overlap_size):
                split_idx = (num_token + self.overlap_size + 1) // 2
                result.append(tokens[:split_idx])
                result.append(tokens[split_idx - self.overlap_size :])
            else:
                for i in range(0, num_token - self.overlap_size, self.stride):
                    result.append(tokens[i : i + self.chunk_size])

            indices.append((start, len(result)))

        return result, indices

    def check_alnum(self, s):
        if len(s) < 2:
            return False
        return not (s.isalpha() or s.isdigit())

    def apply_chunk_merging(self, tokens, next_tokens):
        # Return next tokens if current tokens list is empty
        if not tokens:
            return next_tokens

        source_token_idx = []
        target_token_idx = []
        source_tokens = []
        target_tokens = []
        num_keep = self.overlap_size - self.min_words_cut
        i = 0
        while len(source_token_idx) < self.overlap_size and -i < len(tokens):
            i -= 1
            if tokens[i] not in self.punc_dict:
                source_token_idx.insert(0, i)
                source_tokens.insert(0, tokens[i].lower())

        i = 0
        while len(target_token_idx) < self.overlap_size and i < len(next_tokens):
            if next_tokens[i] not in self.punc_dict:
                target_token_idx.append(i)
                target_tokens.append(next_tokens[i].lower())
            i += 1

        matcher = SequenceMatcher(None, source_tokens, target_tokens)
        diffs = list(matcher.get_opcodes())

        for diff in diffs:
            tag, i1, i2, j1, j2 = diff
            if tag == "equal":
                if i1 >= num_keep:
                    tail_idx = source_token_idx[i1]
                    head_idx = target_token_idx[j1]
                    break
                elif i2 > num_keep:
                    tail_idx = source_token_idx[num_keep]
                    head_idx = target_token_idx[j2 - i2 + num_keep]
                    break
            elif tag == "delete" and i1 == 0:
                num_keep += i2 // 2

        tokens = tokens[:tail_idx] + next_tokens[head_idx:]
        return tokens

    def merge_chunks(self, batch):
        result = []
        if len(batch) == 1 or self.overlap_size == 0:
            for sub_tokens in batch:
                result.extend(sub_tokens)
        else:
            for _, sub_tokens in enumerate(batch):
                try:
                    result = self.apply_chunk_merging(result, sub_tokens)
                except Exception as e:
                    print(e)

        result = " ".join(result)
        return result

    def predict(self, batches):
        t11 = time()
        predictions = []
        for batch, model in zip(batches, self.models):
            batch = util.move_to_device(batch.as_tensor_dict(), 0 if self.device != torch.device("cpu") else -1)
            with torch.no_grad():
                prediction = model.forward(**batch)
            predictions.append(prediction)

        preds, idx, error_probs = self._convert(predictions)
        t55 = time()
        if self.log:
            print(f"Inference time {t55 - t11}")
        return preds, idx, error_probs

    def get_token_action(self, token, index, prob, sugg_token):
        """Get lost of suggested actions for token."""
        # cases when we don't need to do anything
        if prob < self.min_error_probability or sugg_token in [UNK, PAD, '$KEEP']:
            return None

        if sugg_token.startswith('$REPLACE_') or sugg_token.startswith('$TRANSFORM_') or sugg_token == '$DELETE':
            start_pos = index
            end_pos = index + 1
        elif sugg_token.startswith("$APPEND_") or sugg_token.startswith("$MERGE_"):
            start_pos = index + 1
            end_pos = index + 1

        if sugg_token == "$DELETE":
            sugg_token_clear = ""
        elif sugg_token.startswith('$TRANSFORM_') or sugg_token.startswith("$MERGE_"):
            sugg_token_clear = sugg_token[:]
        else:
            sugg_token_clear = sugg_token[sugg_token.index('_') + 1 :]

        return start_pos - 1, end_pos - 1, sugg_token_clear, prob

    def _get_embbeder(self, weights_name, special_tokens_fix):
        embedders = {
            'bert': PretrainedBertEmbedder(
                pretrained_model=weights_name,
                requires_grad=False,
                top_layer_only=True,
                special_tokens_fix=special_tokens_fix,
            )
        }
        text_field_embedder = BasicTextFieldEmbedder(
            token_embedders=embedders,
            embedder_to_indexer_map={"bert": ["bert", "bert-offsets"]},
            allow_unmatched_keys=True,
        )
        return text_field_embedder

    def _get_indexer(self, weights_name, special_tokens_fix):
        if "phobert" in weights_name:
            bert_token_indexer = WordpieceIndexer(
                pretrained_model=weights_name,
                max_pieces_per_token=5,
                do_lowercase=self.lowercase_tokens,
                use_starting_offsets=True,
                special_tokens_fix=special_tokens_fix,
            )
        else:
            bert_token_indexer = PretrainedBertIndexer(
                pretrained_model=weights_name,
                do_lowercase=self.lowercase_tokens,
                max_pieces_per_token=5,
                special_tokens_fix=special_tokens_fix,
            )
        return {'bert': bert_token_indexer}

    def preprocess(self, token_batch):
        seq_lens = [len(sequence) for sequence in token_batch if sequence]
        if not seq_lens:
            return []
        max_len = min(max(seq_lens), self.max_len)
        batches = []
        for indexer in self.indexers:
            batch = []
            for sequence in token_batch:
                tokens = sequence[:max_len]
                tokens = [Token(token) for token in ['$START'] + tokens]
                batch.append(Instance({'tokens': TextField(tokens, indexer)}))
            batch = Batch(batch)
            batch.index_instances(self.vocab)
            batches.append(batch)

        return batches

    def _convert(self, data):
        all_class_probs = torch.zeros_like(data[0]['class_probabilities_labels'])
        error_probs = torch.zeros_like(data[0]['max_error_probability'])
        for output, weight in zip(data, self.model_weights):
            all_class_probs += weight * output['class_probabilities_labels'] / sum(self.model_weights)
            error_probs += weight * output['max_error_probability'] / sum(self.model_weights)

        max_vals = torch.max(all_class_probs, dim=-1)
        probs = max_vals[0].tolist()
        idx = max_vals[1].tolist()
        return probs, idx, error_probs.tolist()

    def update_final_batch(self, final_batch, pred_ids, pred_batch, prev_preds_dict):
        new_pred_ids = []
        total_updated = 0
        for i, orig_id in enumerate(pred_ids):
            orig = final_batch[orig_id]
            pred = pred_batch[i]
            prev_preds = prev_preds_dict[orig_id]
            if orig != pred and pred not in prev_preds:
                final_batch[orig_id] = pred
                new_pred_ids.append(orig_id)
                prev_preds_dict[orig_id].append(pred)
                total_updated += 1
            elif orig != pred and pred in prev_preds:
                # update final batch, but stop iterations
                final_batch[orig_id] = pred
                total_updated += 1
            else:
                continue
        return final_batch, new_pred_ids, total_updated

    def postprocess_batch(self, batch, all_probabilities, all_idxs, error_probs):
        all_results = []
        noop_index = self.vocab.get_token_index("$KEEP", "labels")
        for tokens, probabilities, idxs, error_prob in zip(batch, all_probabilities, all_idxs, error_probs):
            length = min(len(tokens), self.max_len)
            edits = []

            # skip whole sentences if there no errors
            if max(idxs) == 0:
                all_results.append(tokens)
                continue

            # skip whole sentence if probability of correctness is not high
            if error_prob < self.min_error_probability:
                all_results.append(tokens)
                continue

            for i in range(length + 1):
                # because of START token
                if i == 0:
                    token = START_TOKEN
                else:
                    token = tokens[i - 1]
                # skip if there is no error
                if idxs[i] == noop_index:
                    continue

                sugg_token = self.vocab.get_token_from_index(idxs[i], namespace='labels')
                action = self.get_token_action(token, i, probabilities[i], sugg_token)
                if not action:
                    continue

                edits.append(action)
            all_results.append(get_target_sent_by_edits(tokens, edits))
        return all_results

    def handle_batch(self, full_batch, merge_punc=True):
        """
        Handle batch of requests.
        """
        if self.split_chunk:
            full_batch, indices = self.split_chunks(full_batch)
        else:
            indices = None
        final_batch = full_batch[:]
        batch_size = len(full_batch)
        prev_preds_dict = {i: [final_batch[i]] for i in range(len(final_batch))}
        short_ids = [i for i in range(len(full_batch)) if len(full_batch[i]) < self.min_len]
        pred_ids = [i for i in range(len(full_batch)) if i not in short_ids]
        total_updates = 0

        for n_iter in range(self.iterations):
            orig_batch = [final_batch[i] for i in pred_ids]

            sequences = self.preprocess(orig_batch)

            if not sequences:
                break
            probabilities, idxs, error_probs = self.predict(sequences)

            pred_batch = self.postprocess_batch(orig_batch, probabilities, idxs, error_probs)
            if self.log:
                print(f"Iteration {n_iter + 1}. Predicted {round(100*len(pred_ids)/batch_size, 1)}% of sentences.")

            final_batch, pred_ids, cnt = self.update_final_batch(final_batch, pred_ids, pred_batch, prev_preds_dict)
            total_updates += cnt

            if not pred_ids:
                break
        if self.split_chunk:
            final_batch = [self.merge_chunks(final_batch[start:end]) for (start, end) in indices]
        else:
            final_batch = [" ".join(x) for x in final_batch]
        if merge_punc:
            final_batch = [re.sub(r'\s+([\.\,\?\:])', r'\1', x) for x in final_batch]

        return final_batch, total_updates

    def handle_batch_with_metadata(self, full_batch_meta):
        """
        Handle batch of requests and also return metadata after processing.
        """
        full_batch = [[token['text'] for token in batch] for batch in full_batch_meta]
        final_batch, total_updates = self.handle_batch(full_batch, merge_punc=False)
        final_batch_meta = []

        for normalize_text, meta in zip(final_batch, full_batch_meta):
            final_batch_meta.append(self.perfect_matching(meta, normalize_text))

        return final_batch_meta, total_updates

    def perfect_matching(self, text_meta, normalize_text):
        normalize_text_meta = []
        final_text = re.sub(r'\s+([\.\,\?\:])', r'\1', normalize_text)
        # Pattern matching
        if re.search(r'(\d[a-zA-Z]|[a-zA-Z]\d)', normalize_text) is None:
            tokens = final_text.split()
            normalize_text_meta = [
                {
                    'text': tokens[i],
                    'start': text_meta[i]['start'],
                    'end': text_meta[i]['end'],
                }
                for i in range(len(tokens))
            ]
            return normalize_text_meta

        text = " ".join([token['text'] for token in text_meta])

        source_tokens = text.split()
        norm_target_tokens = final_text.split()
        target_tokens = re.sub(r'\s+([\.\,\?\:])', r' ', normalize_text).strip().lower().split()
        matcher = SequenceMatcher(None, source_tokens, target_tokens)
        diffs = list(matcher.get_opcodes())

        for diff in diffs:
            tag, i1, i2, j1, j2 = diff

            if tag == "equal":
                num_target_tokens = j2 - j1
                for c in range(num_target_tokens):
                    normalize_text_meta.append(
                        {
                            'text': norm_target_tokens[j1 + c],
                            'start': text_meta[i1 + c]["start"],
                            'end': text_meta[i1 + c]["end"],
                        }
                    )
            elif tag == "delete":
                normalize_text_meta.extend(text_meta[i1:i2])
            elif tag == "insert":
                warnings.warn("Unexpected insert tag, maybe something wrong happened")
            else:
                start = text_meta[i1]['start']
                end = text_meta[i2 - 1]['end']
                num_target_tokens = j2 - j1
                time_step = (end - start) / num_target_tokens
                for c in range(num_target_tokens):
                    normalize_text_meta.append(
                        {
                            'text': norm_target_tokens[j1 + c],
                            'start': start,
                            'end': start + time_step,
                        }
                    )
                    start += time_step

        return normalize_text_meta
