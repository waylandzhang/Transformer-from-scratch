# -*- coding: utf-8 -*-
"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import Model


# Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed(TORCH_SEED)


encoding = tiktoken.get_encoding("cl100k_base")


# Initiate from trained model
model = Model()
model.load_state_dict(torch.load('model/model-scifi.pt'))
model.eval()
model.to(device)

# start = 'Write a short story about Sam Altman.'
start = 'Sam Altman was born in'
start_ids = encoding.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    y = model.generate(x, max_new_tokens=500)
    print('---------------')
    print(encoding.decode(y[0].tolist()))
    print('---------------')



