'''
We can use the opendatasets library to download the dataset  
https://www.kaggle.com/splcher/animefacedataset) from Kaggle. 
opendatasets uses the Kaggle Official API
for downloading datasets from Kaggle.
   
Follow these steps to find your API credentials:

1. Sign in to  https://kaggle.com,  
then click on your profile picture on the top right and 
select "My Account" from the menu.

2. Scroll down to the "API" section and click 
"Create New API Token". This will download a file `kaggle.json`
 with the following contents:
{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_KEY"}

3. When you run opendatsets.download, you will be asked to 
enter your username & Kaggle API, which you can get from the file 
downloaded in step 2.

Note that you need to download the `kaggle.json` file only once. On Google Colab, you can also upload the `kaggle.json` file using the files tab, and the credentials will be read automatically.

'''
import torch
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import opendatasets as od
import matplotlib.pyplot as plt
import os
from parameters import *
from utils import DeviceDataLoader

print("Downloading the dataset...")
dataset_url = 'https://www.kaggle.com/splcher/animefacedataset'
od.download(dataset_url)


# print(os.listdir(DATA_DIR))
print("Few images from the dataset:")
print(os.listdir(DATA_DIR+'/images')[:10])

train_ds = ImageFolder(DATA_DIR, transform=T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.ToTensor(),
    T.Normalize(*stats)]))

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)

def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
    plt.title("Example of a Batch")
    plt.show()

def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break

show_batch(train_dl)

train_dl = DeviceDataLoader(train_dl, device)









