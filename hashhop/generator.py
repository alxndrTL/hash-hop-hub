from typing import Iterator
import torch
from torch.utils.data import IterableDataset

from hashhop.hashhop import HashHopSampler

class HashHopGenerator(IterableDataset):
    def __init__(self, max_tokens, batch_size, hash_len, max_hops, cot=True, vocab_size=52):
        super().__init__()

        self.sampler = HashHopSampler(max_tokens, hash_len, max_hops, cot, vocab_size)
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            prompt, target = self.sampler.sample(self.batch_size)

            data = torch.cat([prompt, target], dim=1) # (B, L)
            x = data[:, :-1].int() # classic shifting
            y = data[:, 1:].long() # long() is necessary for the CE loss

            yield x, y
