import imp
import torch
import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from parameters import *
# matplotlib.use('TkAgg')  

'''downloading and importing the data as a PyTorch dataset 
using the `MNIST` helper class from `torchvision.datasets`.
'''

mnist = MNIST(root='data', 
              train=True, 
              download=True,
              transform=Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]))

print("Download complete")
print("Example of an Image Tensor from the MNIST Dataset:")
img, label = mnist[0]
print('Label: ', label)
print(img[:,10:15,10:15])
print("Min Value : {} Max Value : {}".format(torch.min(img), torch.max(img)))

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# def raise_window(figname=None):
#     if figname: plt.figure(figname)
#     cfm = plt.get_current_fig_manager()
#     cfm.window.activateWindow()
#     cfm.window.raise_()

# img_norm = denorm(img)
# plt.imshow(img_norm[0], cmap='gray')
# plt.title("Example of an Image Tensor from the MNIST Dataset with Label: {}".format(label))
# plt.show()
# raise_window()

data_loader = DataLoader(mnist, batch_size, shuffle=True)


