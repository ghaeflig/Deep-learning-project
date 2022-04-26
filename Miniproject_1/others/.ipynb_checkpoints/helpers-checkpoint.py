import torch
import os
import matplotlib.pyplot as plt

def psnr(denoised , ground_truth):
	# Peak Signal to Noise Ratio : denoised and ground_truth have range [0, 1]
	mse = torch.mean(( denoised - ground_truth ) ** 2)
	return -10 * torch.log10(mse + 10** -8)


def plot_img_end(denoised, ground_truth, idx=[1,6,10]) :
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
    

    
