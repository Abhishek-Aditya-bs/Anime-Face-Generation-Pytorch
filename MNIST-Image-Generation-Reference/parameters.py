import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt

sample_dir = 'samples'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 100
image_size = 784
hidden_size = 256
latent_size = 64
criterion = nn.BCELoss()

