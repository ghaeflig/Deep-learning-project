import os
import sys
import torch

# leave directory to enable correct imports and savings
from model import *

# Get data and normalize
noisy_imgs_1, noisy_imgs_2 = torch.load('../data/train_data.pkl')
noisy_imgs_1, noisy_imgs_2 = noisy_imgs_1.float()/255, noisy_imgs_2.float()/255
noisy_imgs_1, noisy_imgs_2 = noisy_imgs_1[:15], noisy_imgs_2[:15] #TO REMOVE

# Get run, model and training arguments from terminal
_, run_idx, conv_by_level, pooling_type, batch_norm, dropout, features, optimizer, loss_func  = sys.argv

# architecture
in_channels = out_channels = noisy_imgs_1.shape[1]
conv_by_level = int(conv_by_level)
features = list(map(int, features.strip('[]').split(',')))
pooling_type = pooling_type
batch_norm = bool(int(batch_norm))
dropout = float(dropout)
model_ARGS = [in_channels, out_channels, conv_by_level, features, pooling_type, batch_norm, dropout]

# training
optimizer = optimizer
loss_dict = {'MSE' : nn.MSELoss(), 'MAE' : nn.L1Loss()}
loss_func = loss_dict[loss_func]
batch_size = 50
num_epoch = 10
train_ARGS = [optimizer, loss_func, batch_size, num_epoch]

print(f'Model arguments : \n {model_ARGS} \n {train_ARGS}')

# create unique folder for this run en enter it
path = f'others/run{run_idx}'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

# create model and train it. Best model state will be saved in folder
Noise2noise = Model(model_ARGS, train_ARGS)
losses = Noise2noise.train(noisy_imgs_1, noisy_imgs_2)

# record model loss
torch.save(losses, 'train_val_loss')