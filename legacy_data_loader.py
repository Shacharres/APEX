
import datasets
from transformers import GPT2Tokenizer
import torch


def shakespeare():
    """# **Get the tiny Shakespeare dataset and train on it**"""

    tiny_shake = datasets.load_dataset('text', data_files={'train': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'})['train']

    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

    token_ids = tokenizer.encode([line['text'] for line in tiny_shake], return_tensors="pt").squeeze()
    tokens = torch.tensor(token_ids, dtype=torch.long)
    return tokens


class LegacyDataLoader():
  def __init__(self, data, n_tokens, batch_size):
    self.data = data
    self.n_tokens = n_tokens
    self.batch_size = batch_size
    self.cur_position = 0

  def _next(self):
    while True:
      if self.cur_position + self.n_tokens*self.batch_size + 1 >= len(self.data):
        self.cur_position = 0
        print('resetting the data gen')
      res = self.data[self.cur_position : self.cur_position + self.n_tokens*self.batch_size + 1]
      # reshape to batch*tokens, batch*1 for labels
      x = res[:-1].view(self.batch_size, self.n_tokens)
      y = res[1:].view(self.batch_size, self.n_tokens)
      self.cur_position += self.n_tokens*self.batch_size
      yield (x, y)


class LegacyDataLoaderDDP():
  def __init__(self, data, n_tokens, batch_size, rank=0, world_size=1):
    self.data = data
    self.n_tokens = n_tokens
    self.batch_size = batch_size
    self.rank = rank
    self.world_size = world_size

    self.step = self.n_tokens * self.batch_size
    self.cur_position = self.rank * self.step  # different start per rank

  def _next(self):
    while True:
      if self.cur_position + self.step + 1 >= len(self.data):
        self.cur_position = self.rank * self.step
        if self.rank == 0:
          print('resetting the data gen')

      res = self.data[self.cur_position : self.cur_position + self.step + 1]
      x = res[:-1].view(self.batch_size, self.n_tokens)
      y = res[1:].view(self.batch_size, self.n_tokens)

      # jump by world_size steps so ranks don't overlap
      self.cur_position += self.step * self.world_size
      yield (x, y)