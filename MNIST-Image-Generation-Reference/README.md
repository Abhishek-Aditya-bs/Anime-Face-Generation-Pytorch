# Generating MNSIT Images using GAN'S in Pytorch
Generating MNSIT Images by training a Simple Generative Adversarial Network in PyTorch.

Deep neural networks are used mainly for supervised learning: classification or regression. Generative Adversarial Networks or GANs, however, use neural networks for a very different purpose: Generative modeling

> Generative modeling is an unsupervised learning task in machine learning that involves automatically discovering and learning the regularities or patterns in input data in such a way that the model can be used to generate or output new examples that plausibly could have been drawn from the original dataset. - [Source](https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/)

While there are many approaches used for generative modeling, a Generative Adversarial Network takes the following approach: 

<img src="https://i.imgur.com/6NMdO9u.png" style="width:420px; margin-bottom:32px"/>

There are two neural networks: a *Generator* and a *Discriminator*. The generator generates a "fake" sample given a random vector/matrix, and the discriminator attempts to detect whether a given sample is "real" (picked from the training data) or "fake" (generated by the generator). Training happens in tandem: we train the discriminator for a few epochs, then train the generator for a few epochs, and repeat. This way both the generator and the discriminator get better at doing their jobs. 

GANs however, can be notoriously difficult to train, and are extremely sensitive to hyperparameters, activation functions and regularization.

# Dataset
The dataset is availble in `torchvision.datasets` and the following code will download the dataset

```python
mnist = MNIST(root='data', 
              train=True, 
              download=True,
              transform=Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]))
```

# Simple Discriminator Network

The discriminator takes an image as input, and tries to classify it as "real" or "generated". In this sense, it's like any other neural network. 
Generally a CNN is used for the discriminator, but for this dataset since images have a single channel a simple feedforward network with 
3 linear layers is sufficient. Each 28x28 image is treated as a vector of size 784. Just like any other binary classification model, 
the output of the discriminator is a single number between 0 and 1, which can be interpreted as the probability of the input image being fake 
i.e. generated.

```python
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())
```
# Simple Generator Network
The input to the generator is typically a vector or a matrix which is used as a seed for generating an image. 
Since images have a single channel feedfoward neural network with 3 layers is sufficient.
The output will be a vector of size 784, which can be transformed to a 28x28 px image.

```python
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())
```
# Discriminator Training 
Here are the steps involved in training the discriminator

- The discriminator is expected to output 1 if the image was picked from the real MNIST, and 0 if it was generated using the generator network.??

- Batch of real images is passed into the discriminator, and the loss is computed, by setting the target labels to 1.??

- Then batch of fake images (generated using the generator) is passed into the discriminator, and the loss is computed, by setting the target labels to 0.??

- Finally adding the two losses and using the overall loss to perform gradient descent to adjust the weights of the discriminator.

It's important to note that we don't change the weights of the generator model while training the discriminator (`opt_d` only affects the `discriminator.parameters()`)

# Generator Training
Here are the steps involved in training the Generator

- Batch of fake images is generated using the generator and passed into the discriminator.

- The loss is calculated by setting the target lables tp 1 i.e. real. The reason behind this is to "fool" the discriminator

- Using the loss to perform gradient descent i.e change the weights of the generator, so it gets better at generating real-like images to "fool" the discriminator

# Visualizing Results
<img src="https://github.com/Abhishek-Aditya-bs/Anime-Face-Generation-Pytorch/blob/main/MNIST-Image-Generation-Reference/MNIST-gans_training.gif">
     
# Run the Code
Install `Torch` and `Tqdm` and after cloning the repository run 

```python
python3 main.py
```
### Using a GPU for faster Training 
You can use a [Graphics Processing Unit](https://en.wikipedia.org/wiki/Graphics_processing_unit) (GPU) to train your models faster if your execution platform is connected to a GPU manufactured by NVIDIA. Follow these instructions to use a GPU on the platform of your choice:

* _Google Colab_: Use the menu option "Runtime > Change Runtime Type" and select "GPU" from the "Hardware Accelerator" dropdown.
* _Kaggle_: In the "Settings" section of the sidebar, select "GPU" from the "Accelerator" dropdown. Use the button on the top-right to open the sidebar.
* _Binder_: Notebooks running on Binder cannot use a GPU, as the machines powering Binder aren't connected to any GPUs.
* _Linux_: If your laptop/desktop has an NVIDIA GPU (graphics card), make sure you have installed the [NVIDIA CUDA drivers](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
* _Windows_: If your laptop/desktop has an NVIDIA GPU (graphics card), make sure you have installed the [NVIDIA CUDA drivers](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).
* _macOS_: macOS is not compatible with NVIDIA GPUs

The following function will automatically use a GPU if available else it will train on CPU

```python
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
```

# License
MIT





