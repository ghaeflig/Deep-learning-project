from helpers import *
import sys


_, script_id = sys.argv

if script_id = 1:
    labels = ['baseline', 'w/o batchnorm', 'max pooling', '1-conv layers', 'Adam', 'Adagrad', 'data aug.', 'dropout: 0.5', 'more features']
elif script_id = 2:
    labels = ['baseline', 'max pooling', 'data aug.', 'feat. 1', 'feat. 2', 'feat. 3', 'feat. 4', 'feat. 5']
    
    
create_plot_losses(labels)
create_plot_psnr(labels)