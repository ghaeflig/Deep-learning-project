from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
from functools import reduce
import math
import warnings
import pickle
import time
import random
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

random.seed(0)
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# from torch import set_grad_enabled
# set_grad_enabled(False)

def tensor_dilation2d(tensor, dilation=2):
    """
    Dilates the two last dims of tensor.
    :param tensor: tensor to dilate
    :type tensor: Tensor [... x D2 x D1]
    :param dilation: dilation parameter (dilation-1 zeros introduced between each value), defaults to 2
    :type dilation: int > 0, optional
    :return: dilated tensor
    :rtype: Tensor [... x (D2-1)*dilation+1 x (D1-1)*dilation+1]
    """
    last_dims = [tensor.dim()-2, tensor.dim()-1] # get the two last dims indices
    new_shape = [dilation*(length-1)+1 if id in last_dims else length for id, length in enumerate(tensor.shape)] # create the new_shape for the new_tensor
    new_tensor = empty(new_shape).fill_(0.0).to(device) # fill the new_tensor with 0.0 (defaults value for dilation)
    new_tensor[... , 0::dilation, 0::dilation] = tensor # copy the tensor inside the new_tensor in order to dilate it
    return new_tensor

def tensor_pad2d(tensor, pad=1, only_last_dim=False, only_first_dim=False):
    """
    Pads the two last dims of tensor. only_last_dim and only_first_dim can not be both True at the same time.
    :param tensor: tensor to pad
    :type tensor: Tensor [... x D2 x D1]
    :param pad: pad to add on each of the two last dims, defaults to 1
    :type pad: int >= 0 or list of two ints >=0, optional
    :param only_last_dim: True if padding only the end of the two last dims (asymmetric padding), defaults to False
    :type only_last_dim: bool, optional
    :param only_first_dim: True if padding only the beginning of the two last dims (asymmetric padding), defaults to False
    :type only_first_dim: bool, optional
    :return: padded tensor
    :rtype: Tensor [... x D2+2*pad[0] x D1+2*pad[1]] or [... x D2+pad[0] x D1+pad[1]] if asymmetric padding
    """
    if only_last_dim and only_first_dim:
        warnings.warn("tensor_pad2d: only_last_dim [bool] and only_first_dim [bool] can not be both True at the same time.")
        return None

    # Transform pad variable to a list with 2 int values (one for each dim)
    if type(pad) is int:
        pad = [pad, pad]
    else:
        pad = list(pad)

    # new_shape and new_tensor (filled with 0.0) creations
    new_shape = [length for length in tensor.shape] # new_shape initialization
    if only_last_dim or only_first_dim: # asymmetric padding
        new_shape[-1] += pad[1]
        new_shape[-2] += pad[0]
    else: # symmetric padding
        new_shape[-1] += 2*pad[1]
        new_shape[-2] += 2*pad[0]
    new_tensor = empty(new_shape).fill_(0.0).to(device)

    # indices for the padding
    begin = [pad[0], pad[1]]
    end = [-pad[0], -pad[1]]
    if pad[0] == 0:
        begin[0] = None
        end[0] = None
    if pad[1] == 0:
        begin[1] = None
        end[1] = None

    # adds tensor inside the new_tensor
    if only_last_dim:
        new_tensor[... , :end[0], :end[1]] = tensor
    elif only_first_dim:
        new_tensor[..., begin[0]:, begin[1]:] = tensor
    else:
        new_tensor[..., begin[0]:end[0], begin[1]:end[1]] = tensor
    return new_tensor

def psnr(denoised, ground_truth):
    """
    PSNR (Peak Signal to Noise Ratio) metrics calculations. Function from miniproject 1.
    :param denoised: denoised tensor (values in range [0, 1])
    :type denoised: Tensor [N x C x H x W]
    :param ground_truth: ground truth tensor (values in range [0, 1])
    :type ground_truth: Tensor [N x C x H x W]
    :return: tensors containing the PSNR for each image
    :rtype: Tensor [N]
    """
    # normalise images if needed
    if (ground_truth.max() > 1.0): ground_truth = ground_truth / 255.0
    if (denoised.max() > 20.0): denoised = denoised / 255.0
    mse = ((denoised - ground_truth) ** 2).mean(dim=[1, 2, 3])
    return -10 * (mse + 1e-8).log10()

class Module(object):
    """
    Module base for the implementations of higher-order Modules
    """
    def forward(self, *input):
        """ Forward pass. """
        raise NotImplementedError

    def __call__(self, *input):
        """ Useful for calling the Module forward method more elegantly. """
        return self.forward(*input)

    def backward(self, *gradwrtoutput):
        """ Backward pass. """
        raise NotImplementedError

    def param(self):
        """ Get the Module parameters and their gradients. """
        return []

    def load_param(self, parameters):
        """ Load the parameters given as input. """
        return None

    def test(self, Test):
        """ This function is only used for testing purpose """
        return None

class ReLU(Module):
    """ ReLU layer. """
    def __init__(self):
        pass

    def forward(self, input):
        """ Forward pass of ReLU."""
        output = input.maximum(empty(input.shape).fill_(0.0).to(device))
        return output

    def backward(self, gradwrtoutput):
        """ Backward pass of ReLU."""
        grad = gradwrtoutput * gradwrtoutput.greater(empty(gradwrtoutput.shape).fill_(0.0).to(device)).float()
        return grad

class Sigmoid(Module):
    """ Sigmoid layer. """
    def __init__(self):
        pass

    def forward(self, input):
        """ Forward pass of Sigmoid."""
        output = 1.0 / (1.0 + (-input).exp())
        self.output = output
        return output

    def backward(self, gradwrtoutput):
        """ Backward pass of Sigmoid."""
        grad = gradwrtoutput * self.output * (1.0 - self.output)
        return grad

class Conv2d(Module):
    """
    2D Convolutional layer.
    :param in_channels: input number of channels
    :type in_channels: int > 0
    :param out_channels: output number of channels
    :type out_channels: int > 0
    :param kernel_size: kernel size of the convolution
    :type kernel_size: int > 0
    :param dilation: dilation of the convolution, defaults to 1
    :type dilation: int > 0, optional
    :param padding: padding of the convolution, defaults to 0
    :type padding: int >= 0, optional
    :param stride: stride of the convolution, defaults to 1
    :type stride: int > 0, optional
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        if type(kernel_size) is int: kernel_size = (kernel_size, kernel_size)
        self.kernel_size = tuple(kernel_size)
        self.dilation = int(dilation)
        self.padding = int(padding)
        self.stride = int(stride)
        self.input = None # input (from forward pass) needs to be stored for the backward pass

        # Parameters initialization of torch.nn.Conv2d
        k = math.sqrt(1.0/(self.in_channels * reduce(lambda x, y: x * y, self.kernel_size)))
        self.weight = empty((self.out_channels, self.in_channels) + self.kernel_size).uniform_(-k, k).to(device)
        self.bias = empty((self.out_channels)).uniform_(-k, k).to(device)
        self.grad_weight = empty(self.weight.shape).fill_(0.0).to(device)
        self.grad_bias = empty(self.bias.shape).fill_(0.0).to(device)

    def forward(self, input):
        """
        Forward pass of Conv2d.
        :param input: input tensor
        :type input: Tensor [N x C_in x H_in x W_in]
        :return: output tensor
        :rtype: Tensor [N x C_out x H_out x W_out]
        """
        if input.dim() == 3: input = input.view(1, input.shape[0], input.shape[1], input.shape[2]) # security reshaping
        self.input = input # will be used in the backward pass

        # convolution = unfolding + matrix multiplication + reshaping
        output = unfold(input, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride).to(device)
        output = self.weight.view((self.out_channels, -1)) @ output + self.bias.view((1, -1, 1))

        # shapes (Height, Width) calculations from torch.nn.Conv2d
        H_in = input.shape[2]
        H_out = math.floor((H_in + 2.0 * self.padding - self.dilation * (self.kernel_size[0] - 1.0) - 1.0) / self.stride + 1.0)
        W_in = input.shape[3]
        W_out = math.floor((W_in + 2.0 * self.padding - self.dilation * (self.kernel_size[1] - 1.0) - 1.0) / self.stride + 1.0)

        output = output.view(output.shape[0], output.shape[1], H_out, W_out)
        return output

    def backward(self, gradwrtoutput):
        """
        Backward pass of Conv2d.
        :param gradwrtoutput: gradients with respect to output
        :type gradwrtoutput: Tensor
        :return: gradients with respect to input
        :rtype: Tensor
        """
        if gradwrtoutput.dim() == 3: gradwrtoutput = gradwrtoutput.view(1, gradwrtoutput.shape[0], gradwrtoutput.shape[1], gradwrtoutput.shape[2]) # security reshaping

        # Gradients with respect to input
        dL_dX = gradwrtoutput.clone()
        w = self.weight
        w = w.permute((1, 2, 3, 0))
        w = w.reshape((-1, self.out_channels))
        dL_dX = dL_dX.permute((0, 2, 1, 3))
        dL_dX = w @ dL_dX
        dL_dX = dL_dX.permute((0, 2, 1, 3))
        dL_dX = dL_dX.reshape(dL_dX.shape[0], dL_dX.shape[1], -1)
        output_shape = (self.input.shape[2], self.input.shape[3])
        dL_dX = fold(dL_dX, output_size=output_shape, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride).to(device)

        # Gradients with respect to weights
        PAD_sol = self.dilation*(self.kernel_size[0]-1)-self.input.shape[-1]+self.stride*(gradwrtoutput.shape[-1]-1)+1
        PAD_double = PAD_sol//2
        PAD_simple = PAD_sol%2
        if PAD_sol >= 0:
            i = tensor_pad2d(self.input, pad=PAD_simple, only_first_dim=True)
            i = tensor_pad2d(i, pad=PAD_double)
        else:
            warnings.warn("Conv2d.backward: negative padding")
        dL_dW = unfold(i, kernel_size=gradwrtoutput.shape[-2:], dilation=self.stride, padding=0, stride=self.dilation).to(device)
        dL_dW = dL_dW.reshape(dL_dW.shape[0], self.in_channels, -1, dL_dW.shape[2])
        dL_dW = gradwrtoutput.view(gradwrtoutput.shape[0], 1, gradwrtoutput.shape[1], -1) @ dL_dW
        dL_dW = dL_dW.reshape(dL_dW.shape[0], self.in_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1]).permute((0, 2, 1, 3, 4))
        dL_dW = dL_dW.sum(dim=0)
        self.grad_weight.add_(dL_dW)

        # Gradients with respect to bias
        dL_db = gradwrtoutput.sum(dim=(0,2,3))
        self.grad_bias.add_(dL_db)

        return dL_dX

    def param(self):
        """ Get the Module parameters and their gradients. """
        weight_pair = [self.weight, self.grad_weight]
        bias_pair = [self.bias, self.grad_bias]
        parameters = [weight_pair, bias_pair]
        return parameters

    def load_param(self, parameters):
        """ Load the parameters given as input. """
        self.weight = parameters[0][0]
        self.bias = parameters[1][0]

    def test(self, Test):
        """ This function is only used for testing purpose """
        Test.test__05(self.in_channels, self.out_channels, self.kernel_size[0], self.dilation, self.padding, self.stride)
        Test.test__06(self.in_channels, self.out_channels, self.kernel_size[0], self.dilation, self.padding, self.stride)

class TransposeConv2d(Module):
    """
    2D Transposed Convolutional layer.
    :param in_channels: input number of channels
    :type in_channels: int > 0
    :param out_channels: output number of channels
    :type out_channels: int > 0
    :param kernel_size: kernel size of the convolution
    :type kernel_size: int > 0
    :param dilation: dilation of the convolution, defaults to 1
    :type dilation: int > 0, optional
    :param padding: padding of the convolution, defaults to 0
    :type padding: int >= 0, optional
    :param stride: stride of the convolution, defaults to 1
    :type stride: int > 0, optional
    :param output_padding: additional asymmetric padding parameter (used to match Conv2d-TransposeConv2d shapes), defaults to 0
    :type output_padding: int >= 0, optional
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1, output_padding=0):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        if type(kernel_size) is int: kernel_size = (kernel_size, kernel_size)
        self.kernel_size = tuple(kernel_size)
        self.dilation = int(dilation)
        self.padding = int(padding)
        self.stride = int(stride)
        self.output_padding = int(output_padding)

        self.input = None # input (from forward pass) needs to be stored for the backward pass

        # Parameters initialization of torch.nn.ConvTranspose2d
        k = math.sqrt(1.0 / (self.in_channels * reduce(lambda x, y: x * y, self.kernel_size)))
        self.weight = empty((self.in_channels, self.out_channels) + self.kernel_size).uniform_(-k, k).to(device)
        self.bias = empty((self.out_channels)).uniform_(-k, k).to(device)
        self.grad_weight = empty(self.weight.shape).fill_(0.0).to(device)
        self.grad_bias = empty(self.bias.shape).fill_(0.0).to(device)

    def forward(self, input):
        """
        Forward pass of TransposeConv2d.
        :param input: input tensor
        :type input: Tensor [N x C_in x H_in x W_in]
        :return: output tensor
        :rtype: Tensor [N x C_out x H_out x W_out]
        """
        if input.dim() == 3: input = input.view(1, input.shape[0], input.shape[1], input.shape[2]) # security reshaping
        self.input = input

        output = input.clone()
        w = self.weight
        w = w.permute((1, 2, 3, 0))
        w = w.reshape((-1, self.in_channels))
        output = output.permute((0, 2, 1, 3))
        output = w @ output
        output = output.permute((0, 2, 1, 3))

        H_in = input.shape[2]
        H_out = (H_in - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size[0] - 1) + self.output_padding + 1
        W_in = input.shape[3]
        W_out = (W_in - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size[1] - 1) + self.output_padding + 1

        # This code block is used to correct the output.shape by adding padding if the dimensions are not compatible
        expected_shape_1 = math.floor((H_out + 2 * self.padding - self.dilation * (self.kernel_size[0] - 1) - 1)/self.stride + 1)
        expected_shape_2 = math.floor((W_out + 2 * self.padding - self.dilation * (self.kernel_size[1] - 1) - 1)/self.stride + 1)
        pad_to_add_1 = expected_shape_1 - output.shape[-2]
        pad_to_add_2 = expected_shape_2 - output.shape[-1]
        self.pad_added = [pad_to_add_1, pad_to_add_2]
        if pad_to_add_1 > 0:
            output = tensor_pad2d(output, pad=(pad_to_add_1, 0), only_last_dim=True)
        if pad_to_add_2 > 0:
            output = tensor_pad2d(output, pad=(0, pad_to_add_2), only_last_dim=True)

        output = output.reshape(output.shape[0], output.shape[1], -1)
        output = fold(output, output_size=(H_out, W_out), kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride).to(device)
        output = output + self.bias.reshape(1, -1, 1, 1)

        return output

    def backward(self, gradwrtoutput):
        """
        Backward pass of TransposeConv2d.
        :param gradwrtoutput: gradients with respect to output
        :type gradwrtoutput: Tensor
        :return: gradients with respect to input
        :rtype: Tensor
        """
        if gradwrtoutput.dim() == 3: gradwrtoutput = gradwrtoutput.view(1, gradwrtoutput.shape[0], gradwrtoutput.shape[1], gradwrtoutput.shape[2]) # security reshaping

        # Gradients with respect to input
        dL_dX = unfold(gradwrtoutput, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride).to(device)
        dL_dX = self.weight.view((self.in_channels, -1)) @ dL_dX
        H_in = gradwrtoutput.shape[2]
        H_out = math.floor((H_in + 2.0 * self.padding - self.dilation * (self.kernel_size[0] - 1.0) - 1.0) / self.stride + 1.0)
        W_in = gradwrtoutput.shape[3]
        W_out = math.floor((W_in + 2.0 * self.padding - self.dilation * (self.kernel_size[1] - 1.0) - 1.0) / self.stride + 1.0)
        dL_dX = dL_dX.view(dL_dX.shape[0], dL_dX.shape[1], H_out, W_out)
        dL_dX = dL_dX[:, :, :self.input.shape[2], :self.input.shape[3]] # crops dL_dX to remove non-wanted additional padding added during forward

        # Gradients with respect to weights
        dL_dW = unfold(gradwrtoutput, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride).to(device)
        dL_dW = dL_dW.transpose(0, 1).reshape(self.out_channels * self.kernel_size[0] * self.kernel_size[0], -1).transpose(0, 1)
        dL_dW = self.input.transpose(0, 1).reshape(self.in_channels, -1) @ dL_dW
        dL_dW = dL_dW.view(self.in_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1])
        self.grad_weight.add_(dL_dW)

        # Gradients with respect to bias
        dL_db = gradwrtoutput.sum(dim=(0,2,3))
        self.grad_bias.add_(dL_db)

        return dL_dX

    def param(self):
        """ Get the Module parameters and their gradients. """
        weight_pair = [self.weight, self.grad_weight]
        bias_pair = [self.bias, self.grad_bias]
        parameters = [weight_pair, bias_pair]
        return parameters

    def load_param(self, parameters):
        """ Load the parameters given as input. """
        self.weight = parameters[0][0]
        self.bias = parameters[1][0]

    def test(self, Test):
        """ This function is only used for testing purpose """
        Test.test__07(self.in_channels, self.out_channels, self.kernel_size[0], self.dilation, self.padding, self.stride, self.output_padding)
        Test.test__08(self.in_channels, self.out_channels, self.kernel_size[0], self.dilation, self.padding, self.stride, self.output_padding)

class MSELoss():
    """
    MSE Loss module.
    :param reduction: reduction algorithm (either 'mean' of the losses, either 'sum' of the losses), defaults to 'mean'
    :type reduction: str ('mean' or 'sum'), optional
    """
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, data1, data2):
        return self.forward(data1, data2)

    def forward(self, input, target):
        """
        Forward pass of MSE.
        :param input: tested tensor
        :type input: Tensor [N x C x W x H]
        :param target: target tensor
        :type target: Tensor [N x C x W x H]
        :return: MSE loss
        :rtype: Tensor [1]
        """
        "input and target must have the same shape"
        assert input.shape == target.shape, "MSELoss called with two different-shaped inputs."
        self.input = input
        self.target = target

        loss_n = (input-target) ** 2
        if self.reduction == "mean":
            loss = loss_n.mean()
        elif self.reduction == "sum":
            loss = loss_n.sum()
        else:
            raise Exception("Loss reduction criterion is not defined.")
        return loss

    def backward(self, loss):
        """ Backward pass of MSE. """
        if self.reduction == "mean":
            grad = loss * (2 * self.input - 2 * self.target) / self.input.numel()
        elif self.reduction == "sum":
            grad = loss * (2 * self.input - 2 * self.target)
        return grad

class Sequential():
    """
    Sequential module used to put together the different layers of the model.
    :param modules: ordered list of the modules used in the model
    :type modules: list of Module
    :param forward_initiated: technical parameter only used to check if forward has been called before backward, defaults to False
    :type forward_initiated: bool, optional
    """
    def __init__(self, *modules):
        self.modules = [*modules]
        self.forward_initiated = False

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        """ Forward pass of the model. """
        output = input
        for mod in self.modules:
            output = mod.forward(output)
        self.forward_initiated = True
        return output

    def backward(self, output):
        """ Backward pass of the model. """
        gradwrtoutput = output
        if self.forward_initiated:
            for mod in reversed(self.modules):
                gradwrtoutput = mod.backward(gradwrtoutput)
        else:
            raise Exception("Forward pass must be performed before backward pass.")
        return gradwrtoutput

    def param(self):
        """ Get the parameters of the modules and their gradients. """
        param = []
        for mod in self.modules:
            param += mod.param()
        return param

class SGD():
    """
    Stochastic Gradient Descent optimization.
    Parameters can be obtained from Sequential.param() method -> these parameters are a view (not a copy) of the real parameters from the modules
    :param parameters: parameters to update
    :type parameters: list [len=n_modules] of list [len=n_params/module] of list of 2 Tensors [2]
    :param lr: learning rate
    :type lr: float
    """
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        """ Set all grad Tensors to 0. """
        for i in range(len(self.parameters)):
            self.parameters[i][1].fill_(1.0)

    def step(self):
        """ Update step """
        # self.lr = 0.9*self.lr
        for i in range(len(self.parameters)):
            grad = self.parameters[i][1]
            self.parameters[i][0].add_(self.lr * grad)

class Model():
    """
    Model for miniproject 2.
    """
    def __init__(self):
        # instantiate model (Sequential) + optimizer (SGD) + loss function (MSELoss)
        self.model = Sequential(Conv2d(in_channels=3, out_channels=12, kernel_size=3, dilation=1, padding=1, stride=2),
                                ReLU(),
                                Conv2d(in_channels=12, out_channels=48, kernel_size=3, dilation=1, padding=1, stride=2),
                                ReLU(),
                                TransposeConv2d(in_channels=48, out_channels=12, kernel_size=3, dilation=1, padding=1, stride=2, output_padding=1),
                                ReLU(),
                                TransposeConv2d(in_channels=12, out_channels=3, kernel_size=3, dilation=1, padding=1, stride=2, output_padding=1),
                                Sigmoid())
        self.SGD_optimizer = SGD(self.model.param(), lr=0.1)
        self.MSE = MSELoss()

    def load_pretrained_model(self):
        # This loads the parameters saved in bestmodel.pkl into the model
        self.load_model("bestmodel.pkl") # TODO: add real location of the file

    def save_model(self, filename):
        """ Saves the current model to filename (pkl file). """
        parameters = []
        for mod in self.model.modules:
            parameters.append(mod.param())
        with open(filename, 'wb') as f:
            pickle.dump(parameters, f)

    def load_model(self, filename):
        """ Loads the model of filename (pkl file). """
        with open(filename, 'rb') as f:
            parameters = pickle.load(f)
        for id, mod in enumerate(self.model.modules):
            mod.load_param(parameters[id])

    def shuffle_data(self, train_input, train_target):
        """ Shuffles train_input and train_target in a random order (inplace operation) """
        N = train_input.shape[0]
        order = list(range(N))
        random.shuffle(order)
        train_input = train_input[order]
        train_target = train_target[order]
        return train_input, train_target

    def data_generator(self, train_input, train_target, mini_batch_size=256, shuffle=True):
        """
        Generator of mini batches
        :param train_input: input tensor
        :type train_input: Tensor [N x C x H x W]
        :param train_target: target tensor
        :type train_target: Tensor [N x C x H x W]
        :param mini_batch_size: size of one mini batch, defaults to 256
        :type mini_batch_size: int > 0, optional
        :param shuffle: True if the data need to be shuffled, defaults to True
        :type shuffle: bool, optional
        :return: generator of mini batches (iterator)
        """
        # Copies of the given inputs to avoid to modify the original Tensors
        new_train_input = train_input#.clone()
        new_train_target = train_target#.clone()

        # Shuffles the datasets
        if shuffle:
            new_train_input, new_train_target = self.shuffle_data(new_train_input, new_train_target)

        # Create a generator
        N = train_input.shape[0]
        n_batches = math.ceil(N/mini_batch_size)
        for batch in range(n_batches):
            start = batch * mini_batch_size
            end = min((batch + 1) * mini_batch_size, N)
            if start < end:
                yield new_train_input[start:end], new_train_target[start:end]

    def train(self, train_input, train_target, num_epochs, debug=False):
        """
        Trains the model
        :param train_input: input tensor (containing a noisy version of the images)
        :type train_input: Tensor [N x C x H x W]
        :param train_target: target tensor (containing another noisy version of the same images, which only differs from the input by their noise)
        :type train_target: Tensor [N x C x H x W]
        :param num_epochs: number of epochs (total number of passed through the whole dataset)
        :type num_epochs: int > 0
        :return: None
        """
        # Scales the data to (0.0, 1.0) instead of (0, 255)
        if train_input.max() > 1.0 or train_target.max() > 1.0:
            train_input = train_input / 255.0
            train_target = train_target / 255.0

        t_beginning = time.time() # beginning time of the training
        n_steps = 0
        for epoch in range(num_epochs):
            for input_minibatch, target_minibatch in self.data_generator(train_input, train_target, mini_batch_size=256, shuffle=True):
                predictions = self.model(input_minibatch)
                loss = self.MSE.forward(predictions, target_minibatch)
                n_steps += 1
                if debug and n_steps%10==0: print(f"Loss = {loss} [{epoch},{n_steps}] {round(time.time()-t_beginning,2)} s")
                grad_loss = self.MSE.backward(loss)
                self.SGD_optimizer.zero_grad()
                self.model.backward(grad_loss)
                self.SGD_optimizer.step()

    def predict(self, test_input):
        """
        Predicts the result of test_input (forward pass of the model)
        :param test_input: input tensor with values in range 0-255 that has to be denoised by the trained or the loaded network
        :type test_input: Tensor [N1 x C x H x W]
        :return: prediction tensor with values in range 0-255
        :rtype: Tensor [N1 x C x H x W]
        """
        # Scales the data to (0.0, 1.0) instead of (0, 255)
        if test_input.max() > 1.0:
            test_input = test_input / 255.0

        test_output = self.model(test_input) # Forward pass
        test_output = test_output * 255.0 # Scales back the data to (0, 255)
        return test_output


if __name__ == '__main__':
    train_input, train_target = torch.load("Deep-learning-project/data/train_data.pkl", map_location=device)
    test_noisy, test_cleaned = torch.load("Deep-learning-project/data/val_data.pkl", map_location=device)
    print(train_input.shape, train_target.shape)
    print(test_noisy.shape, test_cleaned.shape)
    print(f"PSNR mean (before) = {psnr(test_noisy, test_cleaned).mean()}")

    # Model training and predictions
    Run = Model()
    Run.train(train_input, train_target, 1)
    test_denoised = Run.predict(test_noisy).int()

    print(f"PSNR mean (after) = {psnr(test_denoised, test_cleaned).mean()}")

    # Quick plotting to observe the result
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(test_noisy[0].permute(1,2,0).to('cpu'))
    ax[1].imshow(test_cleaned[0].permute(1,2,0).to('cpu'))
    ax[2].imshow(test_denoised[0].permute(1,2,0).to('cpu'))
    plt.show()

