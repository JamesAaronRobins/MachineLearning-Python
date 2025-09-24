
import torch

x = torch.rand(5, 3)
print(x)

a = torch.cuda.is_available()
print(a)