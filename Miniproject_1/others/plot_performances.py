import torch
import os
import matplotlib.pyplot as plt

fig, axes = plt.subplots(ncols = 2, figsize = (13,6))

# labels defined with respect to runs defined in the tunings_script.sh
labels = ['baseline', 'w/o batchnorm', 'max pooling', '1-conv layers', 'Adam', 'Adagrad', 'data aug.', 'dropout: 0.5', 'more features']
for i, label in zip(range(1,10,1), labels):
    # load losses
    losses = torch.load(f'run{i}/train_val_loss')
    # plot train and val losses
    axes[0].plot(losses[0], label = label)
    axes[1].plot(losses[1])
    
axes[0].set_title('train loss')
axes[1].set_title('validation loss')
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('loss')
axes[1].set_xlabel('epoch')

fig.suptitle('Comparison of performance for different model settings')
fig.legend()

# make sure not to overwrite previous results
idx = 1
path = f'Performances_{idx}.png'
while os.path.exists(path):
    idx += 1
    path = f'Performances_{idx}.png'

# save figure
fig.savefig(path)
plt.close()