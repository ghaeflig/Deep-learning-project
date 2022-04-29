from helpers import *
import sys


_, script_id = sys.argv

script_id = int(script_id)
if script_id == 0:
    labels = ['baseline', 'w/o batchnorm', 'max pooling', '1-conv layers', 'Adam', 'Adagrad', 'data aug.', 'dropout: 0.5', 'more features']
elif script_id == 1:
    labels = ['baseline', 'max pooling', 'data aug.', 'feat. 1', 'feat. 2', 'feat. 3', 'feat. 4', 'feat. 5']
    
    
create_plot_losses(script_id, labels)
create_plot_psnr(script_id, labels)