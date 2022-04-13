import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, TensorDataset
import time


# ARCHITECTURE classes and functions
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
    

    
    
class Model(nn.Module):
    def __init__(self, model_ARGS, train_ARGS, shape_control = False):
        super(Model, self).__init__()
        
        # Dezip arguments for model architecture and training
        in_channels, out_channels, conv_by_level, features, pooling_type, batch_norm, dropout = model_ARGS
        optimizer, loss_func, batch_size, num_epoch = train_ARGS
        
        # Record class arguments
        self.shape_control = shape_control
        self.optimizer = optimizer # GIVE A STRING ?
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.depth = len(features)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Setting number of CONV PER FLOOR
        if conv_by_level == 1: self.conv_func = Single_Conv
        elif conv_by_level == 2: self.conv_func = Double_Conv
        
        # Setting POOLING TYPE
        if pooling_type == 'max': self.Pooling = nn.MaxPool2d((2,2))
        elif pooling_type == 'average': self.Pooling = nn.AvgPool2d((2,2))
            
        # Creating ENCODING layers
        self.DOWNCONV = nn.ModuleList()
        for feature in features:
            self.DOWNCONV.append(self.conv_func(in_channels, feature, batch_norm, dropout))
            in_channels = feature
        
        # Creating DECODING layers
        self.UPSCALING = nn.ModuleList()
        self.UPCONV = nn.ModuleList()
        for feature in features[::-1]:
            self.UPSCALING.append(nn.ConvTranspose2d(2*feature, feature, kernel_size = 2, stride = 2))
            self.UPCONV.append(self.conv_func(2*feature, feature, batch_norm, dropout))
        
        # Bottom convolution
        self.deepest_conv = self.conv_func(features[-1], features[-1]*2, batch_norm, dropout)
        
        # Final conv
        self.final_layer = nn.Conv2d(features[0], out_channels, kernel_size = (3,3), stride = 1, padding = 1)
            
            
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
        checkpoint = torch.load('others/parameters.pt', map_location=self.device)
        epoch = checkpoint['epoch']
        print("=> Loading checkpoint from a trained model at the best epoch {}".format(epoch))
        self.load_state_dict(checkpoint['model'])
        if self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum = 0.9)
            optimizer.load_state_dict(checkpoint['optimizer'])


    def train(self, train_input, train_target) -> None:
		#: train_input : tensor of size (N, C, H, W) containing a noisy version of the images
		#: train_target : tensor of size (N, C, H, W) containing another noisy version of the same images , which only differs from the input by their noise .
        
        # prepare data for training
        """ ANY PREPROCESSING """
        split_ratio = 0.9
        n = train_input.shape[0]
        shuffled = torch.randperm(train_input.shape[0])
        train_input = train_input[shuffled]
        train_target = train_target[shuffled]
        train_input = train_input[0:int(n*split_ratio),:,:,:]
        train_target = train_target[0:int(n*split_ratio),:,:,:]
        val_input = train_input[int(n*split_ratio):n,:,:,:]
        val_target = train_target[int(n*split_ratio):n,:,:,:]
        
        # prepare model for training
        """problem with function name"""
        #self.train(mode=True)
        if self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum = 0.9)
        
        # keep track of loss
        losses = []
        val_losses = []
        best_loss = 1000
        
        for epoch in range(self.num_epoch):
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
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss = running_loss / len(input_batches)
            print ('Epoch [%d/%d], Train loss: %.4f' %(epoch+1, self.num_epoch, epoch_loss))
            losses.append(epoch_loss)
            
            """# Validation
            val_input_batches = val_input.split(self.batch_size)
            val_target_batches = val_target.split(self.batch_size)
            self.eval()
            with torch.no_grad() :
                for idx, (val_input_batch, val_target_batch) in enumerate(zip(val_input_batches, val_target_batches)):
                    val_pred = self(val_input_batch)
                    val_loss = self.loss_func(val_pred, val_target_batch)
                    val_running_loss += val_loss.item()
            
            val_epoch_loss = val_running_loss / len(val_input_batches)
            print ('Epoch [%d/%d], Validation loss: %.4f' %(epoch+1, self.num_epoch, val_epoch_loss))
            val_losses.append(val_epoch_loss)
            
            
            if val_epoch_loss < best_loss : # à changer avec la val loss plus tard
                best_loss = val_epoch_loss
                best_epoch = epoch
                print("=> Saving checkpoint")
                checkpoint = {'epoch': epoch, 'model': self.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, 'others/parameters.pt')
        
        print('Training finished with best best results at epoch {} | Validation loss : {} | Training loss :'.format(best_epoch, best_loss, losses[epoch]))"""
        
        return losses

    
    
    def predict(self, test_input) -> torch.Tensor:
		#: test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained the loaded network .
		#: returns a tensor of the size (N1 , C, H, W)
        #self.train(False)
        self.eval()
        test_output = torch.empty(test_input.shape)
        test_batches = test_input.split(self.batch_size) #shuffled or not ?
        with torch.no_grad() :
            for b in range(0, test_input.size(0), self.batch_size) :
                out = self(test_batch)
                for k in range(self.batch_size) :
                    test_output[b + k,:,:,:] = out[k,:,:,:]
        return test_output
