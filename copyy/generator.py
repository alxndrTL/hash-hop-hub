import torch
from torch.utils.data import IterableDataset

from copyy.copy import CopySampler

class CopyGenerator(IterableDataset):
    def __init__(self, max_tokens, batch_size, vocab_size=52):
        super().__init__()

        self.sampler = CopySampler(max_tokens, vocab_size)
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            prompt, target = self.sampler.sample(self.batch_size)

            data = torch.cat([prompt, target], dim=1) # (B, L)
            x = data[:, :-1].int() # classic shifting
            y = data[:, 1:].long() # long() is necessary for the CE loss

            yield x, y, prompt.shape[1]
