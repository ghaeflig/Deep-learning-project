import sys
import os
import torch
from helpers import create_imgs_plot, create_best_psnr
sys.path.append('../')
from model import * 

""" HARDCODED PARAMETERS """
batch_size = 50
num_epoch = 10

# set seeds for reproducibility
torch.manual_seed(0)

# Get data and normalize (will be shuffled before training)
noisy_imgs_1, noisy_imgs_2 = torch.load('../../data/train_data.pkl')
noisy_imgs_1, noisy_imgs_2 = noisy_imgs_1.float()/255, noisy_imgs_2.float()/255
noisy_imgs_1, noisy_imgs_2 = noisy_imgs_1[:100], noisy_imgs_2[:100]

# create unique folder for this run en enter it
path = f'best_model'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

# create model and train it. Best model state will be saved in folder
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Noise2noise = Model().to(DEVICE)
losses = Noise2noise.train(noisy_imgs_1, noisy_imgs_2)

# record model loss
torch.save(losses, 'train_val_loss')

# create plot of validation imgs for illustration and psnr performance
print('Creating performance plots...')
noisy_imgs, ground_truth = torch.load('../../../data/val_data.pkl')
noisy_imgs, ground_truth = noisy_imgs.float()/255, ground_truth.float()/255
denoised_imgs = Noise2noise.predict(noisy_imgs)

create_imgs_plot(noisy_imgs, denoised_imgs, ground_truth)
create_best_psnr(denoised_imgs, ground_truth)
print(f'New performance plots available in: {path}')