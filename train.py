"""

main training script.
default is

doesnt support :
-gradient_acc
-multi GPU

when launching a training run, will create a name for the run (wandb run or random if no wandb logging) and save checkpoint, config and final model in runs/{run_name}

if you're using the WSD scheduler and you just want to cooldown a model over N steps, set :
lr_warmup_iters = 0
lr_decay_iters = N
num_iters = N

also, when using the WSD scheduler, a checkpoint will automatically be saved just before the cooldown (independently of ckpt_interval)

"""

import sys
import os
import string
from contextlib import nullcontext
from dataclasses import asdict
import json
import random
import numpy as np
import time
import wandb

import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from utils.lr_schedules import cosine_warmup_schedule, wsd_schedule

from models.lm import LM
from models.transformer.transformer import TransformerConfig
from models.mamba.mamba import MambaConfig
from models.mamba.mamba2 import Mamba2Config

from grokfast import gradfilter_ema

from hashhop.generator import HashHopGenerator
from copyy.generator import CopyGenerator

from eval import eval_hashhop, eval_copy

from utils.misc import format_time

# ---------------------------------------------

seed = 0 # 0, 1, 2...

# --- hash-hop parameters ---
max_tokens = 512
hash_len = 8
max_hops = 3
cot = True
vocab_size = 52

# --- downstream eval parameters ---
eval_interval = 1000
num_tasks = 100

# --- model parameters ---
architecture = "Transformer" # "Transformer" or "Mamba" or "Mamba2"
d_model = 1024
n_layers = 12
bias = False
base_std = 0.02

# Mamba specific
use_cuda = True # choose True if you can (mamba-ssm installed). else, fallbacks to mamba.py (https://github.com/alxndrTL/mamba.py)

# Mamba2 specific
d_head = 54
d_state = 64

# Transformer specific
d_ff = 3584
n_heads = 16
n_kv_heads = n_heads # n_heads is MHA, 1 is MQA (multi query attention), in between is GQA (grouped query attention)
dropout = 0.

pos_emb = "rope" # "absolute" or "rope"
rope_theta = 10000

optimised_attn = False
efficient_attn = False
super_attn = False

use_flash_attention = True

# --- muP parameters ---
use_mup = False
mup_base_width = 288

# --- training parameters ---
num_iters = 100000
batch_size = 32

optimizer = "AdamW" # "AdamW" or "Adam-mini"

# LR and scheduler
schedule = "wsd" # "cosine" or "wsd"

lr = 5e-5
lr_warmup_iters = 1000

# cosine schedule specific
lr_min = 4e-5

# wsd schedule specific
lr_decay_iters = 20000 # 10-20% of num_iters

adam_b1 = 0.9
adam_b2 = 0.95

clip_value_grad = 1.0
weight_decay = 0.1

# grokfast
grokfast_alpha = 0.98
grokfast_lamb = 2

use_torch_compile = True # do not toggle if using Mamba

device = "cuda" # "cpu", "cuda:0", "cuda:1", ...
dtype = "bfloat16" # "float32", "float16" or "bfloat16" (float16 will use a GradScaler)

# --- saving/checkpointing parameters ---
save_dir = "runs/" # where to save to
ckpt_interval = 20000 # None if you don't want checkpointing
# size of each checkpointing file, in MB (if no scaler) : 12 * number of parameters (in M)

ckpt = "" # if you want to restart training from a checkpoint (path/to/model.pth)
start_iter = 0 # specify starting iter (if loading from ckpt_60000, put 60001)

# --- logging and eval parameters ---
log_wandb = True

train_log_interval = 50
eval_val_interval = 200 # also the printing period
eval_val_iters = 50

# --- benchmarking parameters ---
benchmark = False # if set, disables all the extra features (wandb, eval...) and prints benchmarks (time per iter, GPU usage...)

# ---------------------------------------------

seed = 123456789 + seed

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"
torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
dtype_ctx = (nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type, torch_dtype))

if benchmark:
    print("Benchmarking mode enabled.")

    log_wandb = False
    eval_val_interval = 999999
    ckpt_interval = 999999
    eval_interval = 999999

if log_wandb:
    wandb.init(project="hash-hop-hub",
            config={
                "hash-hop": {
                    "max_tokens": max_tokens,
                    "hash_len": hash_len,
                    "max_hops": max_hops,
                    "cot": cot,
                    "vocab_size": vocab_size,
                },
                "model": {
                    "architecture": architecture,
                    "d_model": d_model,
                    "n_layers": n_layers,
                    "bias": bias,
                    "base_std": base_std,
                    # Transformer
                    "d_ff": d_ff,
                    "n_heads": n_heads,
                    "n_kv_heads": n_kv_heads,
                    "dropout": dropout,
                    "pos_emb": pos_emb,
                    "rope_theta": rope_theta,
                    "optimised_attn": optimised_attn or efficient_attn or super_attn,
                    "efficient_attn": efficient_attn or super_attn,
                    "super_attn": super_attn,
                    # Mamba2
                    "d_head_m2": d_head,
                    "d_state_m2": d_state,                    
                },
                "training": {
                    "seed": seed-123456789,
                    "num_iters": num_iters,
                    "batch_size": batch_size,
                    "optimizer": optimizer,
                    "adam_b1": adam_b1,
                    "adam_b2": adam_b2,
                    "clip_value_grad": clip_value_grad,
                    "weight_decay": weight_decay,
                    # lr
                    "schedule": schedule,
                    "lr": lr,
                    "lr_min": lr_min,
                    "lr_warmup_iters": lr_warmup_iters,
                    "lr_decay_iters": lr_decay_iters,
                    # muP
                    "use_mup": use_mup,
                    "mup_base_width": mup_base_width,
                    
                }
            })

if log_wandb:
    run_name = wandb.run.name
else:
    run_name = ''.join(random.choice(string.ascii_letters) for _ in range(8))

if not benchmark:
    save_dir = os.path.join(save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Run name: {run_name}.")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

ds = HashHopGenerator(max_tokens=max_tokens, batch_size=batch_size, hash_len=hash_len, max_hops=max_hops, cot=True)
#ds = CopyGenerator(max_tokens=max_tokens, batch_size=batch_size, vocab_size=vocab_size)
loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=8, pin_memory=True, worker_init_fn=seed_worker, generator=g)
iter_ = iter(loader)

# model
if architecture == "Transformer":
    config = TransformerConfig(d_model=d_model, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads, d_ff=d_ff, pos_emb=pos_emb, rope_theta=rope_theta, base_std=base_std, mup=use_mup, mup_base_width=mup_base_width, optimised_attn=optimised_attn, efficient_attn=efficient_attn, super_attn=super_attn, dropout=dropout, bias=bias, max_len=max_tokens, flash=use_flash_attention)
elif architecture == "Mamba":
    config = MambaConfig(d_model=d_model, n_layers=n_layers, bias=bias, base_std=base_std, mup=use_mup, mup_base_width=mup_base_width, use_cuda=use_cuda)
elif architecture == "Mamba2":
    config = Mamba2Config(d_model=d_model, n_layers=n_layers, d_state=d_state, d_head=d_head, n_groups=1, max_len=max_tokens, bias=bias, base_std=base_std, mup=use_mup, mup_base_width=mup_base_width)
else:
    raise NotImplementedError

model = LM(config, vocab_size=vocab_size+3).to(device) # +3 is : padding, =, \n (respectively tokens 0, 1, 2)

if optimizer == "AdamW":
    optim = model.configure_optimizers(weight_decay, lr, (adam_b1, adam_b2), device_type)
elif optimizer == "Adam-mini": # todo : mup
    raise NotImplementedError
else:
    raise NotImplementedError

scaler = torch.cuda.amp.GradScaler(enabled=(dtype=="float16")) # needed when training with float16

if ckpt != "":
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint["model"])
    optim.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])

if schedule == "cosine":
    scheduler = lr_scheduler.LambdaLR(optim, cosine_warmup_schedule(lr=lr, lr_min=lr_min, warmup_iters=lr_warmup_iters, num_iters=num_iters, start_iter=start_iter))
elif schedule == "wsd":
    scheduler = lr_scheduler.LambdaLR(optim, wsd_schedule(warmup_iters=lr_warmup_iters, decay_iters=lr_decay_iters, num_iters=num_iters, start_iter=start_iter))
else:
    raise NotImplementedError

print(f"Model initialized. Number of parameters : {sum([p.numel() for p in model.parameters()])}.")

unoptimized_model = model # the unoptimized model is kept for saving
if use_torch_compile:
    print("Compiling the model...")
    model = torch.compile(model)
    print("Done compiling.")

print("Training is starting.")
start_time = time.time()
last_time = start_time
last_print_time = start_time

torch.cuda.reset_peak_memory_stats(device=None)

grads = None

try:
    for iter in range(start_iter, num_iters):
        data = next(iter_)
        x, y, prompt_len = data
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with dtype_ctx:
            logits = model(x)[:, prompt_len-1+8:].contiguous() # hashhop
            #logits = model(x)[:, prompt_len-1:].contiguous() # copy
            logits = logits.view(-1, logits.size(-1))
            loss = F.cross_entropy(logits, y[:, prompt_len-1+8:].contiguous().view(-1), ignore_index=0) # hashhop
            #loss = F.cross_entropy(logits, y[:, prompt_len-1:].contiguous().view(-1), ignore_index=0) # copy

        scaler.scale(loss).backward()

        grads = gradfilter_ema(model, grads=grads, alpha=grokfast_alpha, lamb=grokfast_lamb)

        if clip_value_grad != 0.0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value_grad)
        
        # compute grad norm to log it afterwards
        if iter % train_log_interval == 0:
            grad_norms = [p.grad.norm() for p in model.parameters() if p.grad is not None]
            total_grad_norm = torch.sqrt(torch.sum(torch.tensor([gn ** 2 for gn in grad_norms])))
        
        scaler.step(optim)
        scaler.update()

        optim.zero_grad(set_to_none=True)

        # lr decay
        scheduler.step()
        lr_iter = scheduler.get_last_lr()[1] # param group 1 has a "fixed" lr (ie not affected by muP)

        # logging : print and wandb
        to_log = {}
        if iter % train_log_interval == 0:
            to_log.update({"train_loss": loss.item(), "grad_norm": total_grad_norm})

            curr_time = time.time()
            dt = curr_time - last_time
            last_time = curr_time

            time_per_iter = dt / train_log_interval

            to_log.update({"time_per_iter": time_per_iter})

            if benchmark:
                print(f"avg time_per_iter over the last {train_log_interval} iters: {time_per_iter:.5f}s. used GPU memory : {(torch.cuda.memory_allocated(device=None) / (1024**2)):.0f} MB. max used GPU memory : {(torch.cuda.max_memory_allocated(device=None) / (1024**2)):.0f} MB")
                torch.cuda.reset_peak_memory_stats(device=None)
        
        # val loss
        if (iter % eval_val_interval == 0) and not benchmark:
            with torch.no_grad():
                model.eval()
                eval_loss = 0
                for i in range(eval_val_iters):
                    data = next(iter_)
                    x, y, prompt_len = data
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    with dtype_ctx:
                        logits = model(x)[:, prompt_len-1+8:].contiguous()
                        #logits = model(x)[:, prompt_len-1:].contiguous()
                        logits = logits.view(-1, logits.size(-1))
                        loss = F.cross_entropy(logits, y[:, prompt_len-1+8:].contiguous().view(-1), ignore_index=0)
                        #loss = F.cross_entropy(logits, y[:, prompt_len-1:].contiguous().view(-1), ignore_index=0)
                    eval_loss += loss.item()

                eval_loss /= eval_val_iters
                model.train()
            
            to_log.update({"val_loss": eval_loss})
        
        # eval accuracy
        if (iter % eval_interval == 0) and not benchmark:
            with torch.no_grad():
                model.eval()
                model_generate = model.setup_generation(sample=False)

                success = eval_hashhop(model_generate, n_tasks=num_tasks, batch_size=batch_size, max_tokens=max_tokens, max_hops=max_hops, cot=cot, hash_len=hash_len, vocab_size=vocab_size)
                #success = eval_copy(model_generate, n_tasks=num_tasks, batch_size=batch_size, max_tokens=max_tokens, vocab_size=vocab_size)

                model.train()
            
            to_log.update({"success": success})

        if to_log:
            to_log.update({"lr": lr_iter, "tokens_seen": iter*max_tokens*batch_size})

            # printing
            if "val_loss" in to_log:
                num_digits = len(str(num_iters))
                formatted_iter = f"{iter:0{num_digits}d}"

                uptime = int(time.time()-start_time)

                total_time = ((time.time()-last_print_time) * num_iters) / eval_val_interval
                eta = int(total_time - uptime)

                last_print_time = time.time()

                print(f"Iter {formatted_iter}/{num_iters}. train loss : {loss.item():.3f}. valid loss : {eval_loss:.3f}. lr : {lr_iter:.5f}. uptime : {format_time(uptime)}. ETA : {format_time(eta)}.")
            
            # logging
            if log_wandb:
                wandb.log(to_log, step=iter)

        # checkpointing
        if (ckpt_interval and iter % ckpt_interval == 0) or (schedule == "wsd" and (iter == num_iters-lr_decay_iters)):
            
            dirname = f"ckpt_{iter}/"
            if (schedule == "wsd" and (iter == num_iters-lr_decay_iters)):
                print("----- Starting cooldown -----")
                dirname = f"ckpt_{iter}_before_cooldown"

            os.makedirs(os.path.join(save_dir, dirname), exist_ok=True)
            checkpoint = {"model": unoptimized_model.state_dict(),
                          "optimizer": optim.state_dict(),
                          "scaler": scaler.state_dict()}
            torch.save(checkpoint, os.path.join(save_dir, dirname, "model.pth"))
        
except KeyboardInterrupt:
    print("Training interrupted.")

if benchmark:
    sys.exit()

end_time = time.time()
print(f"Training is done. Took {(end_time-start_time)/60:.2f} minutes.")

# saving : config + model checkpoint (model+optim+scaler)
config_dict = asdict(config)

if isinstance(config, TransformerConfig):
    config_dict['architecture'] = "Transformer"
elif isinstance(config, MambaConfig):
    config_dict['architecture'] = "Mamba"
elif isinstance(config, Mamba2Config):
    config_dict['architecture'] = "Mamba2"
else:
    raise NotImplementedError

json.dump(config_dict, open(os.path.join(save_dir, 'config.json'), 'w'))

checkpoint = {"model": unoptimized_model.state_dict(),
              "optimizer": optim.state_dict(),
              "scaler": scaler.state_dict()}
torch.save(checkpoint, os.path.join(save_dir, "model.pth"))

print(f"Successfully saved checkpoint and config in {save_dir}.")

# final logging (some metrics for wandb)
num_params = sum([p.numel() for p in model.parameters()])

to_log = {"num_params": num_params, "num_iters": iter,
          "num_tokens": iter*batch_size*max_tokens,
          "use_torch_compile": use_torch_compile, "use_flash_attn": use_flash_attention, "dtype": dtype}

if log_wandb:
    wandb.log(to_log)
    wandb.finish()
