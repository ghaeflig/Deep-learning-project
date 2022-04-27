import os
import sys
import torch

# leave directory to enable correct imports and savings
sys.path.append('../')
from model import *

""" HARDCODED PARAMETERS """
#subset = 2000
batch_size = 50
num_epoch = 10

# set seeds for reproducibility
torch.manual_seed(0)

# Get data and normalize
noisy_imgs_1, noisy_imgs_2 = torch.load('../data/train_data.pkl')
noisy_imgs_1, noisy_imgs_2 = noisy_imgs_1.float()/255, noisy_imgs_2.float()/255

# Shuffling in case we take a subset
shuffled = torch.randperm(noisy_imgs_1.shape[0])
noisy_imgs_1 = noisy_imgs_1[shuffled]
noisy_imgs_2 = noisy_imgs_2[shuffled]

"""  Possibility to take a subset for tuning  """
""""""""""""""""""""""""""""""""""""""""""""""""
#noisy_imgs_1, noisy_imgs_2 = noisy_imgs_1[:subset], noisy_imgs_2[:subset]
""""""""""""""""""""""""""""""""""""""""""""""""

# Get run, model and training arguments from terminal
_, run_idx, conv_by_level, pooling_type, batch_norm, dropout, features, optimizer, loss_func, data_aug  = sys.argv

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
data_aug = bool(int(data_aug))
train_ARGS = [optimizer, loss_func, batch_size, num_epoch, data_aug]

#print(train_ARGS)
print(f'Model arguments : \n {model_ARGS} \n {train_ARGS}')

# create unique folder for this run en enter it
path = f'others/run{run_idx}'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

# create model and train it. Best model state will be saved in folder
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Noise2noise = Model(model_ARGS, train_ARGS).to(DEVICE)
losses = Noise2noise.train(noisy_imgs_1, noisy_imgs_2)

# record model loss
torch.save(losses, 'train_val_loss')