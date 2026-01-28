# -*- coding: utf-8 -*-
"""
Group1 - Ex1 - Implementing GPT-2 from scratch
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, config.n_embd*3, bias=True)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.config = config
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        batch, n_tokens = x.size()[:2]
        # Calculate Q, K, V
        applied = self.c_attn(x)
        Q, K, V = applied.split(self.config.n_embd, dim=2)

        if False:  # wanna use flash attention
          # Split to heads
          Q = Q.view(batch, n_tokens, self.config.n_head, self.config.n_embd // self.config.n_head)
          K = K.view(batch, n_tokens, self.config.n_head, self.config.n_embd // self.config.n_head)
          V = V.view(batch, n_tokens, self.config.n_head, self.config.n_embd // self.config.n_head)

          Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)  # dim is now (batch, n_head, block_size, n_embd/n_head) so n_head is also a batch dim

          # Calculate attention
          att = Q @ K.transpose(-2, -1) / math.sqrt(self.config.n_embd // self.config.n_head)    # dim is (batch, n_head, block_size, block_size)
          # Add causal attention mask
          att = att.masked_fill(self.bias[:,:,:n_tokens,:n_tokens] == 0, float('-inf'))

          att = F.softmax(att, dim=-1) # work on the embedding
          att = att @ V     # dim is now (batch, n_head, block_size, n_embd / n_head)

          # Concatenate heads
          att = att.transpose(1, 2).contiguous().view(batch, n_tokens, self.config.n_embd)

        att = F.scaled_dot_product_attention(Q, K, V)

        # Multiply by the output projection
        att = self.c_proj(att)

        # return the result
        return att


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, config.n_embd*4, bias=True)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(config.n_embd*4, config.n_embd, bias=True)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=True)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=True)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
            h = nn.ModuleList([Block(self.config) for i in range(self.config.n_layer)]),
            ln_f = nn.LayerNorm(self.config.n_embd, bias=True),
        ))

        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

    def forward(self, idx):
        device = idx.device
        batch, n_tokens = idx.size()

        positions = torch.arange(n_tokens, dtype=torch.long, device=device)

        # generate the input
        x = self.transformer.wte(idx) + self.transformer.wpe(positions)  # dim will be (batch, vocab, embedding) + (tokens, embedding) so broadcast correctly
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'openai-community/gpt2', 'openai-community/gpt2-medium', 'openai-community/gpt2-large', 'openai-community/gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'openai-community/gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'openai-community/gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'openai-community/gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'openai-community/gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
        # print(sd_keys)

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        # print(sd_keys_hf)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoader: # for the edu_fineweb dataset, based on the DataLoaderLite class
    def __init__(self, B, T, process_rank, num_processes, split, master_process: bool):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'} # 'train' or 'val' in file name

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s] # only include files with 'train' or 'val' in the name
        shards = sorted(shards) # sort the files alphabetically
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards # list of file paths
        assert len(shards) > 0, f"no shards found for split {split}" # check that there are some shards
        if master_process:
            print(f"found {len(shards)} shards for split {split}") # print the number of shards found
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        print("resetting the data gen")
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def _next(self): 
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        print("self.current_position: ", self.current_position)
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


def train_me(warmup: bool = False, n_warmup_steps: int = 30):
    losses, val_losses = [], []
    model.train()
    print("train_me() started, warmup: ", warmup)
    for epoch in range(num_epochs):
        for step in range(num_steps):
            t0 = time.time()
            x, y = generator._next()
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            if not warmup or step > n_warmup_steps:
                optimizer.step()
                # print("optimizer.step() completed at step: ", step)
            print(f'step {step}, loss: {loss.item():.4f}, time: {(time.time()-t0)*1000:.2f}ms')
            losses.append(loss.item())
            # move tensors back to cpu to save gpu memory
            x = x.to("cpu")
            y = y.to("cpu")

            if step % 100 == 5: # print val loss every 100 steps, starting at step 5
                val_loss = get_val_loss()
                val_losses.append(val_loss)
                print(f'---- step {step}, val loss: {val_loss:.4f} ----')
        
        save_checkpoint(model, optimizer, epoch, loss)
        
    return losses, val_losses


def save_checkpoint(model, optimizer, epoch, loss):
    # save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'group1_model_ckpt_{datetime.now().strftime("%Y%m%d_%H%M%S")}.checkpoint')


def load_checkpoint(path):
    # loading checkpoint
    model = GPT(GPTConfig())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    # Use model.train() if you're resuming training, or model.eval() for inference
    model.train_me(warmup=True)
    return model, optimizer


def plot_losses(losses, title):
    plt.figure()
    plt.plot(losses[1:])
    plt.xlabel('step')
    plt.ylabel('negative log likelihood loss')

    plt.plot(np.log10(losses))
    plt.xlabel('step')
    plt.ylabel('negative log likelihood loss (log scale)')
    plt.title(title)
    plt.savefig(f'{title}_losses.png')


def get_val_loss(num_steps=20, batch_size=64):
    val_generator = DataLoader(
        B=batch_size,
        T=cfg.block_size,
        process_rank=0,
        num_processes=1,
        split='val',
        master_process=True
    )    
    model.eval()
    
    val_loss = 0.0
    for _ in range(num_steps):
        x, y = val_generator._next()
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            logits = model(x)
            val_loss += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()

    model.train()
    return val_loss / num_steps


if __name__ == "__main__":

    batch_size = 8
    num_epochs = 1
    num_steps = 10
    lr = 1e-4
    output_path = f'group1_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # You can configure colab to run on a T4 GPU for faster generation
    print("device: ", device)

    cfg = GPTConfig(block_size=128)
    print("cfg: ", cfg)
    model = GPT(cfg)
    model.to(device)

    generator = DataLoader(
        B=batch_size,
        T=cfg.block_size,
        process_rank=0,
        num_processes=1,
        split='train',
        master_process=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses, val_losses = train_me()

    # save trained model
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")

    # plot losses
    plot_losses(losses, 'Training Loss')
    plot_losses(val_losses, 'Validation Loss')




# import torch
# import torch.nn.functional as F
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # You can configure colab to run on a T4 GPU for faster generation
# tokens = [15496, 11, 314, 1101, 257, 3303, 2746, 11] # "Hello, I'm a language model,"
# tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
# tokens = tokens.unsqueeze(0).repeat(5, 1) # Generate in a batch of 5. Shape is (5, 8)
# x = tokens.to(device)

# # Move the model to the correct device
# our_gpt_model.to(device)

# k = 20

# print("Generating on device:", device)
# our_gpt_model.eval()
# # generate!
# while x.size(1) < 30: # max_length=30
#     # forward the model to get the logits
#     with torch.no_grad():
#         logits = our_gpt_model(x)
#         # only care about the last token
#         logits = logits[:, -1, :] # Shape (batch, vocab_size)

#         # Implement top-k masking: set non-top-k logits to -inf
#         topk_values, _ = torch.topk(logits, k, dim=-1)
#         kth_largest_value = topk_values[:, -1].unsqueeze(-1) # shape (batch, 1)
#         logits_masked = torch.where(logits < kth_largest_value, torch.full_like(logits, float('-inf')), logits)

#         # now do softmax
#         probs = F.softmax(logits_masked, dim=-1) # Softmax on (batch, vocab_size)

#         # sample according to prob
#         next = torch.multinomial(probs, num_samples=1, seed=42)
#         x = torch.cat((x, next), dim=1)

# print(tokenizer.batch_decode(x))
# [print(s) for s in tokenizer.batch_decode(x)]
