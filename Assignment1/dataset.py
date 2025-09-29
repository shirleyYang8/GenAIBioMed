import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from progen_vocab import Alphabet

# get alphabet
alphabet = Alphabet()

class ProteinDataset(torch.utils.data.Dataset):
    """
    Loading protein sequence dataset
    """
    def __init__(self, path, alphabet, append_eos=True, split="train"):
        self.seqs = []
        self.append_eos = append_eos
        self.alphabet = alphabet
        self.sizes = []
        self.read_data(path, split)
        self.size = len(self.seqs)

    def read_data(self, path, split):
        lines = open(os.path.join(path, f"{split}.txt")).readlines()
        
        for line in lines:    
            # protein sequence
            seq = line.strip()
            tokens = self.alphabet.encode_line(seq, prepend_bos=True, append_eos=True).long()
            self.seqs.append(tokens)
            self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    def __getitem__(self, i):
        self.check_index(i)
        return (self.seqs[i][: (self.sizes[i]-1)], self.seqs[i][1: ])

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]


def collate_tokens(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res


# define collate function
def collate_fn(batch):
    """
    Pads the input sequences in each batch to the same length.
    """
    inputs, targets = zip(*batch)
    
    padded_inputs = collate_tokens(inputs, alphabet.pad())
    padded_targets = collate_tokens(targets, alphabet.pad())
    
    return padded_inputs, padded_targets


def get_dataloader(data_path, split, batch_size, shuffle=True, append_eos=True):
    dataset = ProteinDataset(data_path, alphabet, append_eos=append_eos, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader
