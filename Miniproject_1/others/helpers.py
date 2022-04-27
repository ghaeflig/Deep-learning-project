import torch
import os
import matplotlib.pyplot as plt

sys.path.append('../')
from model import *

def psnr(denoised , ground_truth):
	# Peak Signal to Noise Ratio : denoised and ground_truth have range [0, 1]
	#mse = torch.mean(( denoised - ground_truth ) ** 2)
    mse = ( denoised - ground_truth ) ** 2
	return -10 * torch.log10(mse + 10** -8)


def create_imgs_plot(denoised, ground_truth, idx=[1,6,10]) :
    # Save a figure of concatenated images of denoised and ground truth whose indices are specified by idx
    cimgs = []
    for i in idx : 
        cimg = torch.cat((denoised[i,:,:,:].permute(1,2,0), ground_truth[i,:,:,:].permute(1,2,0)), axis=1)
        cimgs.append(cimg)
        
    fig, ax = plt.subplots(len(idx), figsize=(15,15))
    for j in range(len(idx)) :
        ax[j].imshow(cimgs[j])
    
    k = 1
    path = f'Concatenated_imgs{k}.png'
    while os.path.exists(path):
        k += 1
        path = f'Concatenated_imgs{k}.png'

    # save figure
    fig.savefig(path)
    plt.close()
    

def create_plot_losses():
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

    
def create_plot_psnr():
    #sample = 100
    noisy_test , clean_test = torch.load ('../data/val_data.pkl')
    noisy_test, clean_test = noisy_test.float()/255, clean_test.float()/255
    #noisy_test_sample,  clean_test_sample = noisy_test[:sample], clean_test[:sample]
    
    labels = ['baseline', 'w/o batchnorm', 'max pooling', '1-conv layers', 'Adam', 'Adagrad', 'data aug.', 'dropout: 0.5', 'more features']
    for i, label in zip(range(1,10,1), labels):
        os.chdir(path='others/run{}'.format(i))
        current_model = Model.to("cuda" if torch.cuda.is_available() else "cpu")
        current_model.load_pretrained_model()
        pred_test = current_model.predict(noisy_test)
        metric = psnr(pred_test)
        plt.boxplot(metric)
        
    plt.show()