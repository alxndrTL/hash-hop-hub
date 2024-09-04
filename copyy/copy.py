import math
import string
import torch

class CopySampler:
    def __init__(self, max_tokens, vocab_size=52):
        #self.max_tokens = max_tokens
        self.vocab_size = vocab_size

        self.string_length = (max_tokens - 1) // 2

    def sample(self, batch_size):
        strings = torch.randint(low=2, high=2+52, size=(batch_size, self.string_length))
        answer = torch.cat([torch.ones(batch_size, 1), strings], dim=1)
        return strings, answer

# works with vocab_size<=52
def hh_to_string(tensor):
    mapping = {0: '_', 1: '>'}
    chars = string.ascii_letters

    for i in range(2, 2+52):
        mapping[i] = chars[(i - 2) % len(chars)]
    
    char_list = [mapping[int(val)] for val in tensor.tolist()]
    return ''.join(char_list)
