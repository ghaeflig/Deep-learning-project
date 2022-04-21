import torch
import matplotlib.pyplot as plt

fig, axes = plt.subpltos(ncols = 2, figsize = (14,8))
labels = ['']
for i, label in zip(range(1,10,1), labels):
    #load losses
    losses = torch.load(f'run{i}/train_val_loss')
    axes[0].plot(losses[0], label = label)
    axes[1].plot(losses[1])