import math
import string
import torch

# todo : possibilité d'une ou plusieurs chaines plus grande que n_hops ?
class HashHop:
    def __init__(self, max_tokens, hash_len = 8, n_hops = 2, cot = True, vocab_size=52):
        
        self.max_tokens = max_tokens
        self.hash_len = hash_len

        n_tokens_in_pair = 2 * self.hash_len + 2 # 2 delimiters =,\n

        n_tokens_in_chain = n_tokens_in_pair * n_hops
        self.n_chains = math.floor(self.max_tokens / n_tokens_in_chain)

        self.n_hops = n_hops
        self.cot = cot
        self.vocab_size = vocab_size

        assert self.n_chains > 0, "n_hops is too big and/or max_tokens too small"

    # todo : comment
    def sample(self, batch_size, verbose=False, b=0):

        n_tokens_in_pair = 2 * self.hash_len + 2 # 2 delimiters =,\n
        
        n_lines_chains = self.n_chains*self.n_hops
        max_lines = math.floor(self.max_tokens / n_tokens_in_pair)

        if verbose:
            print(f"number of hash pairs/lines generated in chains : {n_lines_chains}")
            print(f"max number of hash pairs/lines that fit : {max_lines}")
            print(f"number of hash pairs/lines to generate additionally : {max_lines - n_lines_chains}")
            print()

        # generate hashes
        # there is a very very small chance that two hashes in the same batch are the same. not very a problem.
        hashes = torch.randint(low=2, high=2+self.vocab_size, size=(batch_size, self.n_chains, self.n_hops+1, self.hash_len))

        delimiter_1 =  torch.zeros(batch_size, self.n_chains, self.n_hops, 1, dtype=torch.long) # =
        delimiter_2 = torch.ones(batch_size, self.n_chains, self.n_hops, 1, dtype=torch.long) # \n
        A = torch.cat([hashes[:, :, :-1, :], delimiter_1, hashes[:, :, 1:, :], delimiter_2], dim=3) # (B, n_chains, n_hops, 2*hash_len+2)
        A = A.view(batch_size, self.n_chains*self.n_hops, -1) # (B, n_chains*n_hops, 2*hash_len+2)

        n_pairs = max_lines - n_lines_chains
        new_hashes = torch.randint(low=2, high=2+self.vocab_size, size=(batch_size, 2*n_pairs, self.hash_len))

        delimiter_1 = torch.zeros(batch_size, n_pairs, 1, dtype=torch.long)
        delimiter_2 = torch.ones(batch_size, n_pairs, 1, dtype=torch.long)
        new_hashes = torch.cat([new_hashes[:, 0:n_pairs], delimiter_1, new_hashes[:, n_pairs:], delimiter_2], dim=2) # (B, n_pairs, 2*hash_len+2)
        
        A = torch.cat([A, new_hashes], dim=1) # (B, n_chains*n_hops+n_pairs, 2*hash_len+2)

        if verbose:
            print("original hashes:")
            print(hh_to_string(A.view(batch_size, -1)[b]))

        if self.cot:
            delimiter_1 =  torch.zeros(batch_size, self.n_hops+1, 1, dtype=torch.long) # =
            B = torch.cat([hashes[:, 0], delimiter_1], dim=2)
            B = B.view(batch_size, -1)[:, :-1]
        else:
            B = torch.cat([hashes[:, 0, 0], torch.zeros(batch_size, 1, dtype=torch.long), hashes[:, 0, -1]], dim=1)
        
        A = A[:, torch.randperm(A.shape[1])]
        A = A.view(batch_size, -1)

        if verbose:
            print("shuffling")
            print(hh_to_string(A[b]))

            print("chain to find:")
            print(hh_to_string(B[b]))

        return A, B

# works with vocab_size<=52
def hh_to_string(tensor):
    mapping = {0: '=', 1: '\n'}
    chars = string.ascii_letters

    for i in range(2, 2+52):
        mapping[i] = chars[(i - 2) % len(chars)]
    
    char_list = [mapping[int(val)] for val in tensor.tolist()]
    return ''.join(char_list)
