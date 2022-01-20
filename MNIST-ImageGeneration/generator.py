import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from parameters import *
from dataLoader import denorm

"""
The input to the generator is typically a vector or a matrix which is used as
a seed for generating an image. Once again, to keep things simple, 
we'll use a feedfoward neural network with 3 layers, and the output 
will be a vector of size 784, which can be transformed to a 28x28 px image.
"""

G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())

g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)
    
y = G(torch.randn(2, latent_size))
gen_imgs = denorm(y.reshape((-1, 28,28)).detach())

# plt.title("Example of an Latent Random Nosie Tensor from the Generator")
# plt.imshow(gen_imgs[0], cmap='gray');
# plt.show()

G.to(device);