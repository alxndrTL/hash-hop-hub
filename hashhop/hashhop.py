import math
import string
import torch

class HashHop:
    def __init__(self, max_tokens, hash_len = 8, n_hops = 2, cot = True, vocab_size=52):
        
        self.max_tokens = max_tokens
        self.hash_len = hash_len

        n_tokens_in_pair = 2 * self.hash_len + 2 # 2 delimiters =,\n

        n_tokens_in_chain = n_tokens_in_pair * n_hops # todo : possibilité d'une chaine plus grande que n_hops ?
        self.n_chains = math.floor(self.max_tokens / n_tokens_in_chain)

        self.n_hops = n_hops
        self.cot = cot
        self.vocab_size = vocab_size

        print(self.n_chains*self.n_hops+1)

        assert self.n_chains > 0, "n_hops is too big and/or max_tokens too small"

    # todo : how to ensure that no two hashes are the same? (within the 2nd and 3rd dim)
    # todo : comment
    def sample(self, batch_size):
        # generate hashes :
        hashes = torch.randint(low=2, high=2+self.vocab_size, size=(batch_size, self.n_chains, self.n_hops+1, self.hash_len))

        delimiter_1 =  torch.zeros(batch_size, self.n_chains, self.n_hops, 1, dtype=torch.long) # =
        delimiter_2 = torch.ones(batch_size, self.n_chains, self.n_hops, 1, dtype=torch.long) # \n
        A = torch.cat([hashes[:, :, :-1, :], delimiter_1, hashes[:, :, 1:, :], delimiter_2], dim=3) # (B, n_chains, n_hops, 2*hash_len+2)
        A = A.view(batch_size, self.n_chains*self.n_hops, -1) # (B, n_chains*n_hops, 2*hash_len+2)

        A = A[:, torch.randperm(A.shape[1])]
        
        return A.view(batch_size, -1)
    
def hh_to_string(tensor):
    mapping = {
        0: '=',
        1: '\n'
    }
    
    chars = string.ascii_letters
    for i in range(2, 65):
        mapping[i] = chars[(i - 2) % len(chars)]
    
    char_list = [mapping[int(val)] for val in tensor.tolist()]
    return ''.join(char_list)
