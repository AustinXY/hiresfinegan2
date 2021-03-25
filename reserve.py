import torch
import os

gpu_li = [str(i) for i in [0,1,2,3]]

tensor_li = []
for i in gpu_li:
    a = torch.rand(int(1e9)).to('cuda:'+i)
    tensor_li.append(a)

print('reserving gpu:', ', '.join(gpu_li))
while True:
    for i in range(len(gpu_li)):
        for j in range(5):
            tensor_li[i] * tensor_li[i]
