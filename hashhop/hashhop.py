import math
import string
import torch

class HashHop:
    def __init__(self, max_tokens, hash_len = 8, max_hops = 5, cot = True, vocab_size=52):
        """
        HashHop object, use for sampling hash-hop tasks.

        max_tokens  : the max number of tokens in the prompt (input+completion)
        hash_len    : length of every hashes (in chars)
        max_hops    : max number of hops. each example will have a number of hops sampled from [|1, max_hops|]
        cot         : whether to use chain-of-tought or not
        vocab_size  : number of different chars that composed the hashes. 52=2*26=a...zA...Z
        """
        
        self.max_tokens = max_tokens
        self.hash_len = hash_len

        n_tokens_in_pair = 2 * self.hash_len + 2 # 2 delimiters: = and \n

        n_tokens_in_chain = n_tokens_in_pair * max_hops

        if cot:
            n_tokens_completion = hash_len * max_hops+1 + max_hops  # max_hops+1 hashes and max_hops =
        else:
            n_tokens_completion = 2 * hash_len + 1 # 1 delimiter in non-CoT completion: =
        self.n_chains = math.floor((self.max_tokens - n_tokens_completion) / n_tokens_in_chain)

        self.max_hops = max_hops
        self.cot = cot
        self.vocab_size = vocab_size

        assert self.n_chains > 0, "max_hops is too big and/or max_tokens too small"

    def sample(self, batch_size, verbose=False, b=0):
        """
        Samples batch_size different hash-hop tasks and return prompt and target.

        batch_size    : number of elements in the batch
        verbose       : used for debugging. will display the task for element b in the batch
        b             : see verbose arg.
        """

        n_hops = torch.randint(low=1, high=self.max_hops+1, size=(batch_size,))

        # generate all hashes of the chains
        # a "chain" of hashes is just a way to group hashes (will be important below)
        # (there is a very very small chance that two hashes in the same batch are the same. not very a problem.)

        # so the first item (b=0) of the hashes matrix will look like that (for n_hops=2 and n_chains=2):
        #
        # [hash1, hash2, hash3]
        # [Hash1, Hash2, Hash3]
        #
        # where all hashn/Hashn are vectors that stick out of the screen
        # (I've named hashes that are in the second chain with capital H)
        hashes = torch.randint(low=3, high=3+self.vocab_size, size=(batch_size, self.n_chains, self.max_hops+1, self.hash_len))
        
        # here, we create the unshuffled version of the prompt.
        # what we do is concat two versions (w/ shifting) of the hashes matrix to create A, that will have the form:
        #
        # [hash1, hash2] . [hash2, hash3] = [hash1 = hash2 \n, hash2 = hash3 \n]
        # [Hash1, Hash2]   [Hash2, Hash3]   [Hash1 = Hash2 \n, Hash2 = Hash3 \n]
        #
        # (only b=0 shown)
        # where . is the concatenation operation along the third dim (the one that goes through the screen)
        # (it's concatenation as well as adding the right delimiters)
        # (the delimiters here are shown in their string version (= and \n) but are encoded as 0 and 1 respectively)
        
        delimiter_1 =  torch.ones(batch_size, self.n_chains, self.max_hops, 1, dtype=torch.long) # =
        delimiter_2 = 2*torch.ones(batch_size, self.n_chains, self.max_hops, 1, dtype=torch.long) # \n
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
        # more precisely, the goal is to reconstruct the end of the first chain (for each element)
        # it is not fixed, the number of hops is specified in the n_hops vector
        if self.cot:
            # the goal of the next 3 lines is to select from the hashes matrix the last N hashes of the 1st chain from each element in the batch
            # where N is specified by n_hops (one N per element)
            # remember hashes of shape (B, n_chains, max_hops+1, hash_len) and of the form:
            #
            # [hash1, hash2, hash3]
            #
            # (only b=0 displayed, with n_chains=1 as we are interested in the first one only)

            # to prepare, we first pad hashes with extra 0's along the seq dim (1) so that every chain part is of same length
            B = torch.cat([hashes[:, 0], torch.zeros(batch_size, (self.max_hops+1)-(n_hops.min()+1), self.hash_len, dtype=torch.long)], dim=1)

            # here, we specify which columns from hashes will be selected for each element in the batch
            # the starting index is : length (max_hops+1) - (N+1)
            # and we add to that arange(max_hops+1) to get all the following elements including and after that index
            cols = ((self.max_hops+1) - (n_hops.unsqueeze(1)+1)) + torch.arange(self.max_hops+1)

            # finally, we select the columns
            B = B[torch.arange(B.size(0)).unsqueeze(1), cols] # (B, max_hops+1, hash_len)

            # B now has the form:
            #
            # [hash1_1, hash1_2, hash1_3,     pad,     pad,     pad]
            # [hash2_1, hash2_2,     pad,     pad,     pad,     pad]
            # [hash3_1, hash3_2, hash3_3, hash3_4, hash3_5, hash3_6]
            #
            # (this time, the batch dim is displayed with B=3) (max_hops+1=6)
            # (N=[2, 1, 5])
            # (pad is just 0, and will be part of the prompt target)

            # we're getting there
            # what we need is add the = delimiter between all the hashes of the same batch, similarly to what's been done before
            # however, that's a bit more tricky because simply adding the = delimiter between all hashes would yield something like in the end:
            #
            # hash1_1=hash1_2=hash1_3=pad=pad=pad
            #
            # but we simply want :
            #
            # hash1_1=hash1_2=hash1_3padpadpadpadpadpad
            #
            # so we have to come with a delimiters vector that is = as long as there are hashes, and then pad.
            # that is done below.
            delimiters = torch.zeros(batch_size, self.max_hops+1, 1, dtype=torch.long)
            column_indices = torch.arange(self.max_hops+1).unsqueeze(0).expand(batch_size, self.max_hops+1)
            mask = column_indices < n_hops.unsqueeze(1)
            delimiters[mask] = 1

            B = torch.cat([B, delimiters], dim=2)
            B = B.view(batch_size, -1)[:, :-1]

        else:
            # similar to the CoT case (and easier : no padding)
            cols = torch.cat([(self.max_hops+1-1-n_hops.unsqueeze(1)), torch.tensor([self.max_hops+1-1]).expand(batch_size, 1)], dim=1)
            B = hashes[torch.arange(hashes.size(0)).unsqueeze(1), 0, cols] # (B, 2, hash_len)

            delimiter_1 = torch.ones(batch_size, 2, 1, dtype=torch.long)
            B = torch.cat([B, delimiter_1], dim=2) # (B, 2, hash_len+1)
            B = B.view(batch_size, -1)[:, :-1]

        if verbose:
            print("shuffling")
            print(hh_to_string(A[b]))

            print(f"chain to find (n_hops={n_hops[b]}):")
            print(hh_to_string(B[b]))

        return A, B

# works with vocab_size<=52
def hh_to_string(tensor):
    mapping = {0: '_', 1: '=', 2: '\n'}
    chars = string.ascii_letters

    for i in range(3, 3+52):
        mapping[i] = chars[(i - 2) % len(chars)]
    
    char_list = [mapping[int(val)] for val in tensor.tolist()]
    return ''.join(char_list)
