import torch
import matplotlib.pyplot as plt

fig, axes = plt.subpltos(ncols = 2, figsize = (14,8))
labels = ['']
for i, label in zip(range(1,10,1), labels):
    # load losses
    losses = torch.load(f'run{i}/train_val_loss')
    # plot train and val losses
    axes[0].plot(losses[0], label = label)
    axes[1].plot(losses[1], label = label)
    
axes[0].set_title('train loss')
axes[1].set_title('validation loss')
axes[0].set_xlabel(epoch)
axes[0].set_ylabel(loss)
axes[1].set_xlabel(epoch)

fig.legend