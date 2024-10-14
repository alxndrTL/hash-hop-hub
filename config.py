"""
default config is 124M Transformer
"""

from models.transformer.transformer import TransformerConfig
from models.mamba.mamba import MambaConfig
from models.mamba.mamba2 import Mamba2Config

# ---------------------------------------------

seed = 1 # 0, 1, 2...

# 512, 8, 3, true
# 256, 4, 2, true

# --- hash-hop parameters ---
max_tokens = 256
hash_len = 8
max_hops = 3
cot = True
vocab_size = 52

# --- downstream eval parameters ---
eval_interval = 200
num_tasks = 100

# --- model parameters ---
architecture = "Transformer" # "Transformer" or "Mamba" or "Mamba2"
d_model = 768
n_layers = 12
base_std = 0.02

# Mamba specific
use_cuda = True # choose True if you can (mamba-ssm installed). else, fallbacks to mamba.py (https://github.com/alxndrTL/mamba.py)

# Mamba2 specific
bias = False
d_head = 64
d_state = 128

# Transformer specific
d_ff = 2048
n_heads = 12
n_kv_heads = 12 # n_heads is MHA, 1 is MQA (multi query), in between is GQA (grouped query attention)
dropout = 0.
diff_transformer = True

pos_emb = "rope" # "absolute" or "rope"
rope_theta = 10000

use_flash_attention = True

# --- muP parameters ---
use_mup = False
mup_base_width = 768

# --- training parameters ---
num_iters = 4768 # 9536 = 5B tokens, 4768 = 2.5B tokens, with tbs=512 # 2500 = 36M tokens with bs=14
total_batch_size = 512
micro_batch_size = 32 # 16 for width=768, 32 for width=64

# LR and scheduler
schedule = "wsd" # "cosine" or "wsd"

lr = 3e-4
lr_warmup_iters = 200

# cosine schedule specific
lr_min = lr/10

# wsd schedule specific
lr_decay_iters = 1000 # 10-20% of num_iters

optimizer = "AdamW" # "AdamW" or "Ademamix" or "AdamWScheduleFree"

weight_decay = 0.1
adam_b1 = 0.9
adam_b2 = 0.95

# Ademamix specific (we set T_alpha_beta3 to T)
adam_b3 = 0.997 # heuristic: half of informations comes from last ln(0.5)/ln(b3) gradients
alpha = 5

max_grad_norm = 1.0

use_torch_compile = False # do not toggle if using Mamba

device = "cuda" # "cpu", "cuda:0", "cuda:1", ...
dtype = "bfloat16" # "float32" or "bfloat16"

# --- saving/checkpointing parameters ---
save_dir = "runs/" # where to save to
ckpt_interval = 10000 # None if you don't want checkpointing
# size of each checkpointing file, in MB : 12 * number of parameters (in M)

ckpt = "" # if you want to restart training from a checkpoint (path/to/model.pth)
start_iter = 0 # specify starting iter (if loading from ckpt_60000, put 60001)

# --- logging and eval parameters ---
log_wandb = True

train_log_interval = 12
eval_val_interval = 12 # also the printing period
eval_val_iters = 50

# ---------------------------------------------

if architecture == "Transformer":
    config = TransformerConfig(d_model=d_model, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads, d_ff=d_ff, diff_transformer=diff_transformer, pos_emb=pos_emb, rope_theta=rope_theta, base_std=base_std, mup=use_mup, mup_base_width=mup_base_width, dropout=dropout, max_len=max_tokens, flash=use_flash_attention)
elif architecture == "Mamba":
    config = MambaConfig(d_model=d_model, n_layers=n_layers, bias=bias, base_std=base_std, mup=use_mup, mup_base_width=mup_base_width, use_cuda=use_cuda)
elif architecture == "Mamba2":
    config = Mamba2Config(d_model=d_model, n_layers=n_layers, d_state=d_state, d_head=d_head, n_groups=1, max_len=max_tokens, bias=bias, base_std=base_std, mup=use_mup, mup_base_width=mup_base_width)
else:
    raise NotImplementedError
