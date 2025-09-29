import math
from typing import Dict, Optional, Tuple, Sequence, List, Union
import itertools
import uuid

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter


token2dict = {'F': 10, 'C': 7, 'N': 17, 'Q': 20, 'Z': 29, 'P': 19, 'Y': 28, '<|eos|>': 2, 'B': 6, 'D': 8, 'O': 18,
              'R': 21, 'V': 25, 'G': 11, 'X': 27, 'H': 12, 'L': 15, 'K': 14, '2': 4, 'A': 5, 'S': 22, 'I': 13,
              'M': 16, 'T': 23, 'E': 9, 'W': 26, 'U': 24, '1': 3, '<|bos|>': 1, '<|pad|>': 0}


class Alphabet(object):
    def __init__(
        self,
        prepend_toks: Sequence[str] = ("<null_0>", "<null_2>"),
        prepend_bos: bool = True,
        append_eos: bool = True,
        use_msa: bool = False,
    ):
        self.prepend_toks = list(prepend_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.use_msa = use_msa

        self.all_toks = list(self.prepend_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}
        for token in token2dict:
            self.tok_to_idx[token] = token2dict[token] + 2
        
        self.idx_to_tok = {}
        for tok in self.tok_to_idx:
            self.idx_to_tok[self.tok_to_idx[tok]] = tok
        
        for idx in range(2, len(self.idx_to_tok)):
            self.all_toks.append(self.idx_to_tok[idx])

        self.unk_idx = self.tok_to_idx["<null_0>"]
        self.padding_idx = self.get_idx("<|pad|>")
        self.cls_idx = self.get_idx("<|bos|>")
        self.eos_idx = self.get_idx("<|eos|>")
        self.all_special_tokens = ["<null_0>", "<null_2>", '<|pad|>', '<|bos|>', '<|eos|>']
        self.unique_no_split_tokens = self.all_toks

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.padding_idx

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.cls_idx

    def eos(self):
        return self.eos_idx

    def unk(self):
        return self.unk_idx

    def to_dict(self):
        return self.tok_to_idx.copy()

    def get_batch_converter(self, truncation_seq_length: int = None):
        BatchConverter(self, truncation_seq_length)

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        """
        Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
        Converts a string in a sequence of tokens, using the tokenizer.
        Args:
            text (:obj:`str`):
                The sequence to be encoded.
        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in text]

    def encode_line(
        self,
        line,
        add_if_not_exist=True,
        consumer=None,
        prepend_bos=True,
        append_eos=True,
    ) -> torch.IntTensor:
        tokens = [aa for aa in line]
        ntokens = len(tokens)
        ids = torch.IntTensor(ntokens + int(prepend_bos) + int(append_eos))
        ids.fill_(self.padding_idx)

        if prepend_bos:
            ids[0] = self.cls_idx

        seq_encoded = self.encode(tokens)
        seq = torch.tensor(seq_encoded, dtype=torch.int64)
        ids[int(prepend_bos): len(seq_encoded) + int(prepend_bos)] = seq

        if append_eos:
            ids[len(seq_encoded) + int(prepend_bos)] = self.eos_idx

        return ids

    def string(
            self,
            tensor,
            bpe_symbol=None,
            escape_unk=False,
            extra_symbols_to_ignore=None,
            unk_string=None,
            include_eos=False,
            separator="",
    ):
        extra_symbols_to_ignore = set(self.all_special_tokens)
        extra_symbols_to_ignore.add(self.eos())
        extra_symbols_to_ignore.add(self.bos())
        extra_symbols_to_ignore.add(self.unk())
        extra_symbols_to_ignore.add(self.get_idx("<null_1>"))
        extra_symbols_to_ignore.add(self.pad())

        def token_string(i):
            return self.get_tok(i)

        sent = separator.join(
            token_string(i)
            for i in tensor
            if i.item() not in extra_symbols_to_ignore
        )
        return sent


class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet, truncation_seq_length: int = None):
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list = zip(*raw_batch)
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
        if self.truncation_seq_length:
            seq_encoded_list = [seq_str[:self.truncation_seq_length] for seq_str in seq_encoded_list]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, (label, seq_str, seq_encoded) in enumerate(
            zip(batch_labels, seq_str_list, seq_encoded_list)
        ):
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx

        return labels, strs, tokens
    