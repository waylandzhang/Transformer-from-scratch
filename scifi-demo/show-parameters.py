import torch
from model import Model

model = Model()
state_dict = torch.load('model/model-scifi.pt')
model.load_state_dict(state_dict)

# Calculate the number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"模型参数量为： {total_params:,}")
