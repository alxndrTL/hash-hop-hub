import math
import torch

from hashhop.hashhop import HashHopSampler, hh_to_string

def eval(model_generate, n_tasks, batch_size, max_tokens, max_hops, cot, hops=None, hash_len=8, vocab_size=52, verbose=False):

    sampler = HashHopSampler(max_tokens, hash_len, max_hops, cot, vocab_size)

    n_batches = math.ceil(n_tasks / batch_size)

    success = 0
    for _ in range(n_batches):
        prompts, targets, n_hops = sampler.sample(batch_size, hops, return_hops=True)

        prompts = torch.cat([prompts, targets[:, :hash_len]], dim=1)
        completions = model_generate(prompts, num_tokens=targets.size(1)-hash_len)
        full_answers = [torch.cat([targets[i, :hash_len], completions[i]], dim=0) for i in range(batch_size)]

        results = [torch.allclose(targets[i, :(n_hops[i]+1)*hash_len+1], full_answers[i][:(n_hops[i]+1)*hash_len+1]) for i in range(batch_size)]

        success += sum(results)
        
        # useful for showing the model errors
        if verbose:
            for i in range(batch_size):
                if results[i] == False:
                    print(hh_to_string(prompts[i]))
                    print(hh_to_string(targets[i]))
                    print(hh_to_string(completions[i]))

                    print("-------------------")

    success = success / (n_batches * batch_size)

    return success

# todo : specific n_hops ?
