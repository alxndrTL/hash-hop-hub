import math
import torch

class HashHop:
    def __init__(self, n_tokens, hash_len: int = 8, n_hops = 2, cot = True):
        
        self.n_tokens = n_tokens
        self.hash_len = hash_len

        n_tokens_in_pair = 2 * self.hash_len + 2 # 2 delimiters =,
        #n_hashes_pairs = math.floor(self.n_tokens / n_tokens_in_pair)

        n_tokens_in_chain = n_tokens_in_pair * n_hops # todo : possibilité d'une chaine plus grande que n_hops ?
        self.n_chains = self.n_tokens / n_tokens_in_chain

        self.n_hops = n_hops
        self.cot = cot

    def sample(self):
        # generate hashes
        hashes = torch.randint(low=0, high=128, size=(self.n_hops, self.hash_len)) #self.n_chains)) # todo : 128=vocab_size

        A = torch.cat([hashes[:-1, :], torch.zeros(self.n_hops-1, 1, dtype=torch.long), hashes[1:, :], torch.ones(self.n_hops-1, 1, dtype=torch.long)], dim=1)
        return A.flatten()
