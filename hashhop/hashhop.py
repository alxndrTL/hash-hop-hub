import math
import string
import torch

# todo : comment (args)
# todo : comments (shapes)
# todo : décaler n_hops dans sample
# todo : possibilité d'une ou plusieurs chaines plus grande que n_hops ?
class HashHop:
    def __init__(self, max_tokens, hash_len = 8, max_hops = 5, cot = True, vocab_size=52):
        
        self.max_tokens = max_tokens
        self.hash_len = hash_len

        n_tokens_in_pair = 2 * self.hash_len + 2 # 2 delimiters = and \n

        n_tokens_in_chain = n_tokens_in_pair * max_hops
        self.n_chains = math.floor(self.max_tokens / n_tokens_in_chain)

        self.max_hops = max_hops
        self.cot = cot
        self.vocab_size = vocab_size

        assert self.n_chains > 0, "n_hops is too big and/or max_tokens too small"

    def sample(self, batch_size, n_hops, verbose=False, b=0):
        """
        Samples B hash-hop tasks and return prompt and target.
        """

        assert n_hops <= self.max_hops, "cannot sample tasks with bigger number of hops that the set max_hops"

        n_lines_chains = self.n_chains*self.max_hops
        n_tokens_in_pair = 2 * self.hash_len + 2 # 2 delimiters = and \n
        max_lines = math.floor(self.max_tokens / n_tokens_in_pair)

        if verbose:
            print(f"number of hash pairs/lines generated in chains : {n_lines_chains}")
            print(f"max number of hash pairs/lines that fit : {max_lines}")
            print(f"number of hash pairs/lines to generate additionally : {max_lines - n_lines_chains}")
            print()

        # generate all hashes of the chains
        # a "chain" of hashes is just a way to group hashes (will be important below)
        # (there is a very very small chance that two hashes in the same batch are the same. not very a problem.)

        # so the first item (b=0) of the hashes matrix will look like that (for n_hops=2 and n_chains=2):

        # [hash1, hash2, hash3]
        # [Hash1, Hash2, Hash3]

        # where all hashn/Hashn are vectors that stick out of the screen
        # (I've named hashes that are in the second chain with capital H)
        hashes = torch.randint(low=2, high=2+self.vocab_size, size=(batch_size, self.n_chains, self.max_hops+1, self.hash_len))
        
        # here, we create the unshuffled version of the prompt.
        # what we do is concat two version of the hashes matrix to create A, that will have the form:

        # [hash1, hash2] . [hash2, hash3] = [hash1 = hash2 \n, hash2 = hash3 \n]
        # [Hash1, Hash2]   [Hash2, Hash3]   [Hash1 = Hash2 \n, Hash2 = Hash3 \n]

        # (only b=0 shown)
        # where . is the concatenation operation along the third dim (the one that goes through the screen)
        # (it's concatenation as well as adding the right delimiters)
        # (the delimiters here are shown in their string version (= and \n) but are encoded as 0 and 1 respectively)
        
        delimiter_1 =  torch.zeros(batch_size, self.n_chains, self.max_hops, 1, dtype=torch.long) # =
        delimiter_2 = torch.ones(batch_size, self.n_chains, self.max_hops, 1, dtype=torch.long) # \n
        A = torch.cat([hashes[:, :, :-1, :], delimiter_1, hashes[:, :, 1:, :], delimiter_2], dim=3) # (B, n_chains, n_hops, 2*hash_len+2)

        # with what we have from above, we simply have to flatten to get something very similar to the unshuffled version of the prompt.
        # this will leave us with A:
        # [hash1 = hash2 \n, hash2 = hash3 \n, Hash1 = Hash2 \n, Hash2 = Hash3 \n]
        # (only b=0 shown)
        # (again, keep in mind that hash1 = hash2 \n is a vector that goes through the screen)
        A = A.view(batch_size, self.n_chains*self.max_hops, -1) # (B, n_chains*n_hops, 2*hash_len+2)

        if verbose:
            print("original hashes:")
            print(hh_to_string(A.view(batch_size, -1)[b]))

        # finally, shuffling
        A = A[:, torch.randperm(A.shape[1])]
        A = A.view(batch_size, -1)

        # here, we create the target
        # the target chain to reconstruct is always the first chain (or at least part of it)
        if self.cot:
            # in the CoT case, we concat the last n_hops+1 hashes of the first chain (as well as adding the = delimiter)
            delimiter_1 =  torch.zeros(batch_size, n_hops+1, 1, dtype=torch.long) # =
            B = torch.cat([hashes[:, 0, -(n_hops+1):], delimiter_1], dim=2)
            B = B.view(batch_size, -1)[:, :-1]
        else:
            # in the no-CoT case, we just concat the -(n_hops+1) and last hash of the first chain
            # (as well as adding the = delimiter)
            delimiter_1 = torch.zeros(batch_size, 1, dtype=torch.long)
            B = torch.cat([hashes[:, 0, -(n_hops+1)], delimiter_1, hashes[:, 0, -1]], dim=1)

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
