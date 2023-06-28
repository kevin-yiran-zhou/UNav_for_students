import torch

a=torch.rand((3,10))
for i in a:
    print(sum(i<0.5))