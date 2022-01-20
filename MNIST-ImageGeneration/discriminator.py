import torch.nn as nn
import torch
from parameters import *


""" 
We use the Leaky ReLU activation for the discriminator.
Different from the regular ReLU function, 
Leaky ReLU allows the pass of a small gradient signal for negative values. 
As a result, it makes the gradients from the discriminator flows 
stronger into the generator. Instead of passing a gradient (slope) of 0 
in the back-prop pass, it passes a small negative gradient.

Just like any other binary classification model, the output of the discriminator 
is a single number between 0 and 1, which can be interpreted as the probability 
of the input image being fake i.e. generated.
"""

D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
D.to(device)