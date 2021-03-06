import torch
import os
import matplotlib.pyplot as plt
import sys

sys.path.append('../')
from model import *

def psnr(denoised , ground_truth):
	# Peak Signal to Noise Ratio : denoised and ground_truth have range [0, 1]
	#mse = torch.mean(( denoised - ground_truth ) ** 2)
    # normalise images if needed
    if (torch.max(ground_truth) > 1): ground_truth = ground_truth/255
    if (torch.max(denoised) > 20): denoised = denoised/255
        
    mse = torch.mean((denoised - ground_truth)**2, dim=[1,2,3])
    return -10 * torch.log10(mse + 10**-8)


def create_imgs_plot(noisy, denoised, ground_truth, idx=[1,6,10]) :
    # Save a figure of concatenated images of denoised and ground truth whose indices are specified by id     
    
    # Make sure images are in the correct range [0, 255]
    if torch.max(noisy) <= 20 : noisy = noisy * 255
    if torch.max(denoised) <= 20 : denoised = denoised * 255
    if torch.max(ground_truth) <= 20 : ground_truth = ground_truth * 255
            
    # Make sure images are integers to enable correct visualization
    noisy = noisy.type(torch.uint8)
    denoised = denoised.type(torch.uint8)
    ground_truth = ground_truth.type(torch.uint8)
    
    # plot
    fig, ax = plt.subplots(len(idx), 3, figsize=(15,17))
    for j, i in enumerate(idx) :
            ax[j, 0].imshow(noisy[i,:,:,:].permute(1,2,0))
            ax[j, 0].axis('off')
            ax[j, 1].imshow(denoised[i,:,:,:].permute(1,2,0))
            ax[j, 1].axis('off')
            ax[j, 2].imshow(ground_truth[i,:,:,:].permute(1,2,0))
            ax[j, 2].axis('off')
            
    ax[0,0].set_title('Noisy images', fontsize=30)
    ax[0,1].set_title('Denoised images', fontsize=30)
    ax[0,2].set_title('Ground-truth', fontsize=30)
            
    plt.suptitle('Noisy, denoised and ground-truth images for some samples of the validation data', fontsize=20)
    save_figure('Concatenated_imgs')
    plt.close()
    

def create_plot_losses(script_id, labels):
    # adjust folders index to corresponding script
    offset = 9*script_id
    
    #plot figure
    fig, axes = plt.subplots(ncols = 2, figsize = (13,6))
    
    # labels defined with respect to runs defined in the tunings_script.sh
    for i, label in zip(torch.arange(1,10,1)+offset, labels):
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

    save_figure('losses')
    plt.close()    

    
def create_plot_psnr(script_id, labels):
    # adjust folders index to corresponding script
    offset = 9*script_id
    
    noisy_test, clean_test = torch.load('../../data/val_data.pkl')
    #noisy_test_sample,  clean_test_sample = noisy_test[:sample], clean_test[:sample]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    psnrs = []
    means = []
    for i, label in zip(torch.arange(1,10,1)+offset, labels):
        os.chdir(path='run{}'.format(i))
        
        # get model
        arg = torch.load('bestmodel.pth', map_location=device)
        model_ARGS = [arg['in_channels'], arg['conv_by_level'], arg['features'],
                      arg['pooling_type'], arg['batch_norm'], arg['dropout']]
        train_ARGS = [arg['optimizer'], arg['loss_func'], arg['batch_size'], arg['data_aug']]
        
        current_model = Model(model_ARGS, train_ARGS).to(device)
        current_model.load_pretrained_model()
        pred_test = current_model.predict(noisy_test)
        metric = psnr(pred_test, clean_test)
        psnrs.append(metric)
        means.append(torch.mean(metric))
        os.chdir(path='../')
        
    bp = plt.boxplot(psnrs, labels = labels, showmeans = True)
    plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'])
    plt.title('Boxplots of psnr over validation images for each trained model')
    plt.xticks(rotation = 45)
    plt.rc('xtick', labelsize=2)
    save_figure('models_psnr')
    plt.close()
    
    # create txt with means in numerical values
    with open("PSNR_MEANS.txt", "w") as file:
        for label, mean in zip(labels, means):
            file.write(f"{label}: mean = {mean}\n")
    
def create_best_psnr(denoised, ground_truth):
    metric = psnr(denoised, ground_truth)
    bp = plt.boxplot(metric.numpy(), showmeans = True)
    plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'])
    plt.text(1.1, 31, f'mean: {round(torch.mean(metric).item(),2)}\nmedian: {round(torch.median(metric).item(),2)}')
    plt.title('Boxplot of psnr (metric performance) of the best model on validation data')
    save_figure('best_model_psnr_boxplot')
    plt.close()

    
def save_figure(path):
    """ Save figure at <path> without overwriting if exists """
    k = 1
    final_path = f'{path}_{k}.png'
    while os.path.exists(final_path):
        k += 1
        final_path = f'{path}_{k}.png'
    plt.savefig(final_path)