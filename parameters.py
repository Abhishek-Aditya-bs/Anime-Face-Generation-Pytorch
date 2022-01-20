from utils import get_default_device

lr = 0.0002
epochs = 25
image_size = 64
batch_size = 128
latent_size = 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
DATA_DIR = './animefacedataset'
sample_dir = 'generated'
device = get_default_device()