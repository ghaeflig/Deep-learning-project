import torch
import random
import torchvision.transforms.functional as TF
from torchvision.transforms import GaussianBlur

random.seed(0)
torch.manual_seed(0)

def load_train_data(path = '../../data/train_data.pkl', data_aug = True) :
    train_input, train_target = torch.load(path)

    if (torch.max(train_input) > 1 and torch.max(train_target) > 1) :
        train_input, train_target = train_input.float()/255, train_target.float()/255

        # Data augmentation
        if data_aug :
            #Data augmentation : horizontal flip
            print('Data augmentation...')
            id_hflip = random.sample(range(0, train_input.shape[0]), int(train_input.shape[0]/2))
            img_hflip = TF.hflip(train_input[id_hflip,:,:,:])
            target_hflip = TF.hflip(train_target[id_hflip,:,:,:])

            #Data augmentation : gaussian blurr one image of the pairs
            id_gaus = random.sample(range(0, train_input.shape[0]), int(train_input.shape[0]/2))
            img_gaus = train_input[id_gaus,:,:,:]
            target_gaus = GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))(train_target[id_gaus,:,:,:])
            
            #Concatenation with original data
            train_input = torch.cat((train_input, img_hflip, img_gaus), 0)
            train_target = torch.cat((train_target, target_hflip, target_gaus), 0)
        
        # shuffle dataset
        shuffled = torch.randperm(train_input.shape[0])
        input_shuffled = train_input[shuffled]
        target_shuffled = train_target[shuffled]
        
        # Exchange image pairs to avoid depedency of the model on input noise
        v = torch.randn(train_input.shape[0]) > 0
        target_shuffled[v,:,:,:], input_shuffled[v,:,:,:] = input_shuffled[v,:,:,:], target_shuffled[v,:,:,:]
        
        return input_shuffled, target_shuffled


def load_test_data(path = '../../data/val_data.pkl') :
    noisy_img, cleaned_img = torch.load(path)
    return noisy_img, cleaned_img