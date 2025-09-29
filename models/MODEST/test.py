import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, current_dir)

from Model import ISGNet
import torch
import time

model = ISGNet()
model.to("cuda")
rgb_img = torch.rand(1, 3, 384, 384).to("cuda")
depth_img = torch.rand(1, 1, 384, 384).to("cuda")
final = model(rgb_img, depth_img)
print(type(final))

while True: 
    rgb_img = torch.rand(1, 3, 384, 384).to("cuda")
    depth_img = torch.rand(1, 1, 384, 384).to("cuda")
    start_time = time.time()
    final = model(rgb_img, depth_img)
    print(f"It took {time.time() - start_time}s to complete")
    time.sleep(5)