import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import GaussianBlur
#from torch.utils.data import DataLoader, TensorDataset
import time
import random
import sys

# set seed for reproducibility
random.seed(0)
torch.manual_seed(0)

######################################
# ARCHITECTURE classes and functions #
######################################

class Single_Conv(nn.Module):
    """ Implements 1 convolution by level. Optional batch_norm and dropout """
    def __init__(self, in_c, out_c, batch_norm, dropout):
        super(Single_Conv, self).__init__()
    
        conv1 = nn.Conv2d(in_c, out_c, kernel_size = (3,3), stride = 1, padding = 1)
        if not batch_norm:
            self.conv = nn.Sequential(conv1, nn.ReLU())
        else:
            BN = nn.BatchNorm2d(out_c)
            self.conv = nn.Sequential(conv1, BN, nn.ReLU())
        
        # Setting DROPOUT probability
        self.drop_proba = dropout
        self.drop = nn.Dropout2d(p=dropout)
        
    def forward(self, img):
        img = self.conv(img)
        if self.drop_proba != 0:
            return self.drop(img)
        return img
    
    
class Double_Conv(nn.Module):
    """ Implements 2 convolutions by level. Optional batch_norm and dropout """
    def __init__(self, in_c, out_c, batch_norm, dropout):
        super(Double_Conv, self).__init__()
        
        conv1 = nn.Conv2d(in_c, out_c, kernel_size = (3,3), stride = 1, padding = 1)
        conv2 = nn.Conv2d(out_c, out_c, kernel_size = (3,3), stride = 1, padding = 1)
        if not batch_norm:
            self.conv = nn.Sequential(conv1, nn.ReLU(), conv2, nn.ReLU())
        else:
            BN = nn.BatchNorm2d(out_c)
            self.conv = nn.Sequential(conv1, BN, nn.ReLU(), conv2, BN, nn.ReLU())
    
    # Setting DROPOUT probability
        self.drop_proba = dropout
        self.drop = nn.Dropout2d(p=dropout)
        
    def forward(self, img):
        img = self.conv(img)
        if self.drop_proba != 0:
            return self.drop(img)
        return img
    


######################################
#             Model class            #
######################################

    
class Model(nn.Module):
    def __init__(self, model_ARGS = None, train_ARGS = None, shape_control = False):
        super(Model, self).__init__()
        
        if model_ARGS == None:
            """ Hardcoded best model architecture corresponding to pretrained model (instantiated if no arguments are given) """
            # architecture arguments
            self.in_channels = self.out_channels = 3
            self.conv_by_level = 1
            self.pooling_type = 'average'
            self.batch_norm = False
            self.dropout = 0
            self.features = [16,32,64]
            
            # train arguments (wont be used unless training is launched)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.shape_control = False
            self.optimizer = 'SGD'
            self.loss_func = nn.MSELoss().to(self.device)
            self.batch_size = 100
            self.depth = len(self.features)
            self.data_aug = True
            
        else:
            """ Instantiate model with arguments for experimenting """
            # architecture arguments
            self.in_channels, self.conv_by_level, self.features, self.pooling_type, self.batch_norm, self.dropout = model_ARGS
            self.out_channels = self.in_channels
            
            # train arguments
            self.optimizer, self.loss_func, self.batch_size, self.data_aug = train_ARGS
            self.shape_control = shape_control
            self.depth = len(self.features)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            
            
        
        # Setting number of CONV PER FLOOR
        if self.conv_by_level == 1: self.conv_func = Single_Conv
        elif self.conv_by_level == 2: self.conv_func = Double_Conv
        
        # Setting POOLING TYPE
        if self.pooling_type == 'max': self.Pooling = nn.MaxPool2d((2,2))
        elif self.pooling_type == 'average': self.Pooling = nn.AvgPool2d((2,2))
            
        # Creating ENCODING layers
        self.DOWNCONV = nn.ModuleList()
        in_channels = self.in_channels
        for feature in self.features:
            self.DOWNCONV.append(self.conv_func(in_channels, feature, self.batch_norm, self.dropout))
            in_channels = feature
        
        # Creating DECODING layers
        self.UPSCALING = nn.ModuleList()
        self.UPCONV = nn.ModuleList()
        for feature in self.features[::-1]:
            self.UPSCALING.append(nn.ConvTranspose2d(2*feature, feature, kernel_size = 2, stride = 2))
            self.UPCONV.append(self.conv_func(2*feature, feature, self.batch_norm, self.dropout))
        
        # Bottom convolution
        self.deepest_conv = self.conv_func(self.features[-1], self.features[-1]*2, self.batch_norm, self.dropout)
        
        # Final conv
        self.final_layer = nn.Conv2d(self.features[0], self.out_channels, kernel_size = (3,3), stride = 1, padding = 1)
            
            
    def forward(self, x):
        # Optional shape tracking for control
        if self.shape_control: print(f'Original shape: {x.shape}')
        
        # Record pooled layers for skip connections in decoding
        skip_conn = []
        
        # ENCODING
        for idx, layer in enumerate(self.DOWNCONV):
            # convolution
            conv_x = layer(x)
            skip_conn.append(conv_x)
            
            # pooling
            x = self.Pooling(conv_x)
            if self.shape_control: print(f'Level {idx+1}: conv shape: {conv_x.shape}, pooled shape: {x.shape}')
        
        # DEEPEST CONV
        x = self.deepest_conv(x)
        if self.shape_control: print(f'Deepest shape: {x.shape}')
        
        # Flip skip connections for concatenation in decoding
        skip_conn = skip_conn[::-1]
        
        # DECODING
        for idx, (upscale_layer, conv_layer, connection) in enumerate(zip(self.UPSCALING, self.UPCONV, skip_conn)):
            # upscaling
            up_x = upscale_layer(x)

            # handle odd number for shape
            if up_x.shape != connection.shape:
                up_x = TF.resize(up_x, size=connection.shape[2:])
            
            # concatenate with skip connection
            concat_x = torch.cat((connection,up_x), dim=1)
            
            # convolution
            x = conv_layer(concat_x)
            if self.shape_control: print(f'Level {self.depth-idx}: upscaled shape: {up_x.shape}, concat shape: {concat_x.shape}, conv shape: {x.shape}')

        x = self.final_layer(x)
        if self.shape_control: print(f'Final shape: {x.shape}')
        return x
        
        
        
    def load_pretrained_model(self) -> None:
		## This loads the parameters saved in bestmodel .pth into the model
        #checkpoint = torch.load('bestmodel.pth', map_location=self.device)
        checkpoint = torch.load('Deep-learning-project/Miniproject_1/bestmodel.pth', map_location=self.device)
        epoch = checkpoint['epoch']
        print("=> Loading checkpoint from a trained model at the best epoch {}".format(epoch))
        self.load_state_dict(checkpoint['model_state'])
        self.set_optimizer()
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])


    def train(self, train_input, train_target, num_epoch=50) : 
        
        """ Prepare data for training """
        # Preprocessing
        if (torch.max(train_input) > 1 and torch.max(train_target) > 1) :
            train_input, train_target = train_input.float()/255, train_target.float()/255

        # Data augmentation
        if self.data_aug :
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
        
        # info on device state before starting training
        if not torch.cuda.is_available() :
            print("\nThings will go much quicker if you enable a GPU \n")
        else :
            print("\nYou are running the training of the data on a GPU \n")
        
        # keep track of a small validation loss
        split_ratio = 0.8
        n = train_input.shape[0]
        train_input = input_shuffled[0:int(n*split_ratio),:,:,:].to(self.device)
        train_target = target_shuffled[0:int(n*split_ratio),:,:,:].to(self.device)
        val_input = input_shuffled[int(n*split_ratio):n,:,:,:].to(self.device)
        val_target = target_shuffled[int(n*split_ratio):n,:,:,:].to(self.device)
        
        # prepare model for training
        self.set_optimizer()
        
        # keep track of loss
        train_losses = []
        val_losses = []
        best_loss = sys.maxsize
        
        for epoch in range(num_epoch):
            self.train_func()
            time_begin = time.time()
            
            # shuffle data to avoid overfitting and create batches
            input_batches = train_input.split(self.batch_size)
            target_batches = train_target.split(self.batch_size)
            
            running_loss = 0
            val_running_loss = 0
            
            # Train on each batch
            for idx, (input_batch, target_batch) in enumerate(zip(input_batches, target_batches)):
                # time info
                if (idx+1)%5 == 0:
                    print(f'Batch {idx+1} : {round(time.time()-time_begin,2)} sec')
                pred = self(input_batch)
                loss = self.loss_func(pred, target_batch)
                running_loss += loss.item()

                # Backward pass and gradient update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # EVALUATE TRAINING loss with mean over batch losses
            epoch_loss = running_loss / len(input_batches)
            train_losses.append(epoch_loss)
            
            # EVALUATE VALIDATION loss with mean over unseen batch
            val_input_batches = val_input.split(self.batch_size)
            val_target_batches = val_target.split(self.batch_size)
            
            self.eval_func()
            with torch.no_grad() :
                for idx, (val_input_batch, val_target_batch) in enumerate(zip(val_input_batches, val_target_batches)):
                    val_pred = self(val_input_batch)
                    val_loss = self.loss_func(val_pred, val_target_batch)
                    val_running_loss += val_loss.item()
                    
                    
            
            val_epoch_loss = val_running_loss / len(val_input_batches)
            print ('Epoch [%d/%d], Train loss: %.4f' %(epoch+1, num_epoch, epoch_loss), 'Validation loss: %.4f' %val_epoch_loss)
            val_losses.append(val_epoch_loss)
            
            # UPDATE BEST MODEL given the validation loss
            if val_epoch_loss < best_loss :
                best_loss = val_epoch_loss
                best_epoch = epoch
                print("=> Saving checkpoint")
                # record all aspects of model for correct best model architecture hardcoding and parameters loading later
                checkpoint = {'epoch': epoch+1, 'model_state': self.state_dict(), 'optimizer': self.optimizer,
                              'optimizer_state': self.optimizer.state_dict(), 'batch_norm' : self.batch_norm,
                              'in_channels' : self.in_channels, 'conv_by_level' : self.conv_by_level,
                              'pooling_type' : self.pooling_type, 'features' : self.features, 'dropout' : self.dropout,
                              'loss_func' : self.loss_func, 'batch_size' : self.batch_size, 'data_aug' : self.data_aug}
                torch.save(checkpoint, 'bestmodel.pth')
        
        print('Training finished with best best results at epoch {} | Training loss : {:.4f} | Validation loss : {:.4f} '.format(best_epoch+1, train_losses[best_epoch], best_loss))
        
        return train_losses, val_losses

    
    
    def predict(self, test_input) -> torch.Tensor:
        self.eval_func()
        # normalize and put on device
        normalize = False
        if torch.max(test_input) > 1 :
            normalize = True
            test_input = test_input.float()/255
        test_input = test_input.to(self.device)
        
        # if the test input can not be split into batches
        if test_input.shape[0] < self.batch_size :
            return self(test_input)

        # prepare for prediction
        test_output = torch.empty(test_input.shape)
        test_batches = test_input.split(self.batch_size)
        
        with torch.no_grad() :
            for idx, test_batch in enumerate(test_batches):
                out = self(test_batch)
                for k in range(self.batch_size) :
                    test_output[idx*self.batch_size + k,:,:,:] = out[k,:,:,:]
        # denormalized output            
        #if normalize : test_output = test_output*255           
        return test_output
    
    
    def set_optimizer(self) -> None:
        " Called when the model is used to set the optimizer, as it cannot be done before __init__() is done. "
        if self.optimizer == 'SGD': self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum = 0.9)
        elif self.optimizer == 'Adam': self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))
        elif self.optimizer == 'Adagrad': self.optimizer = torch.optim.Adagrad(self.parameters(), lr=0.01, lr_decay=0, weight_decay=0)
        # If the optimizer is not defined by a string, it is already loaded correctly
    
    def train_func(self, mode=True) -> torch.Tensor:
        """ Sets model in training mode (original source code, because of imposed train() class function overwrites it) """
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval_func(self) -> torch.Tensor:
        """ Sets model in evaluation mode (original source code, because of imposed train() class function overwrites it) """
        return self.train_func(False)
    
    
        