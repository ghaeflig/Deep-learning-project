import os
import torch

os.chdir('../')
print(os.getcwd())
from model import *
print('pass')

exit()

""" HARDCODED PARAMETERS """
batch_size = 50
num_epoch = 10

# set seeds for reproducibility
torch.manual_seed(0)

# Get data and normalize
noisy_imgs_1, noisy_imgs_2 = torch.load('../data/train_data.pkl')
noisy_imgs_1, noisy_imgs_2 = noisy_imgs_1.float()/255, noisy_imgs_2.float()/255

# create unique folder for this run en enter it
path = f'others/best_model'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

# create model and train it. Best model state will be saved in folder
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Noise2noise = Model().to(DEVICE)
losses = Noise2noise.train(noisy_imgs_1, noisy_imgs_2)

# record model loss
torch.save(losses, 'train_val_loss')

