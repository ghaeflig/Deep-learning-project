import torch
import os
import matplotlib.pyplot as plt
import sys

sys.path.append('../')
from model import *

def psnr(denoised , ground_truth):
	# Peak Signal to Noise Ratio : denoised and ground_truth have range [0, 1]
	#mse = torch.mean(( denoised - ground_truth ) ** 2)
    mse = torch.mean((denoised - ground_truth)**2, dim=[1,2,3])
    return -10 * torch.log10(mse + 10**-8)


def create_imgs_plot(noisy, denoised, ground_truth, idx=[1,6,10]) :
    # Save a figure of concatenated images of denoised and ground truth whose indices are specified by id     
        #cimg = torch.cat((noisy[i,:,:,:].permute(1,2,0), denoised[i,:,:,:].permute(1,2,0), ground_truth[i,:,:,:].permute(1,2,0)), axis=1)
        #cimgs.append(cimg)
        
    fig, ax = plt.subplots(len(idx), 3, figsize=(15,17))
    for j, i in enumerate(idx) :
            ax[j, 0].imshow(noisy[i,:,:,:].permute(1,2,0))
            ax[j, 0].axis('off')
            ax[j, 1].imshow(denoised[i,:,:,:].permute(1,2,0))
            ax[j, 1].axis('off')
            ax[j, 2].imshow(ground_truth[i,:,:,:].permute(1,2,0))
            ax[j, 2].axis('off')
            
    ax[0,0].set_title('Noisy images')
    ax[0,1].set_title('Denoised images')
    ax[0,2].set_title('Ground-truth images')
            
    plt.suptitle('Noisy, denoised and ground-truth images for some samples of the validation data')
    
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
    path = f'losses_{idx}.png'
    while os.path.exists(path):
        idx += 1
        path = f'losses_{idx}.png'

    # save figure
    fig.savefig(path)
    plt.close()    

    
def create_plot_psnr():
    #sample = 100
    noisy_test, clean_test = torch.load('../../data/val_data.pkl')
    noisy_test, clean_test = noisy_test.float()/255, clean_test.float()/255
    #noisy_test_sample,  clean_test_sample = noisy_test[:sample], clean_test[:sample]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    psnrs = []
    labels = ['baseline', 'w/o batchnorm', 'max pooling', '1-conv layers', 'Adam', 'Adagrad', 'data aug.', 'dropout: 0.5', 'more features']
    for i, label in zip(range(1,10,1), labels):
        os.chdir(path='run{}'.format(i))
        
        arg = torch.load('bestmodel.pth', map_location=device)
        model_ARGS = [arg['in_channels'], arg['conv_by_level'], arg['features'], arg['pooling_type'], arg['batch_norm'], 0]#arg['dropout']]
        train_ARGS = [arg['optimizer'], arg['loss_func'], arg['batch_size'], arg['num_epoch'], 0]#arg['data_aug']]
        
        current_model = Model(model_ARGS, train_ARGS).to(device)
        current_model.load_pretrained_model()
        pred_test = current_model.predict(noisy_test)
        metric = psnr(pred_test, clean_test)
        psnrs.append(metric)
        os.chdir(path='../')
        
    bp = plt.boxplot(psnrs, labels = labels, showmeans = True)
    plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'])
    plt.title('Boxplots of psnr over validation images for each trained model')
    plt.xticks(rotation = 45)
    plt.rc('xtick', labelsize=2)
    plt.savefig('models_psnr.png')
    plt.close()
    
    
def create_best_psnr(denoised, ground_truth):
    metric = psnr(denoised, ground_truth)
    bp = plt.boxplot(metric.numpy(), showmeans = True)
    plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'])
    plt.title('Boxplot of psnr (metric performance) of the best model on validation data')
    plt.savefig('best_model_psnr_boxplot.png')
    plt.close()
    