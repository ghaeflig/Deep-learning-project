from torch import empty, cat, arange
from torch.nn.functional import fold, unfold
from functools import reduce
import math

# from torch import set_grad_enabled
# set_grad_enabled(False)
# from torch import Tensor


"""
This is the first draft of miniproject 2. It is not a stable release and some errors are not corrected.
Some functions are still under development: ConvTranspose / Upsampling / ...
Debugging/unit testing in progress.
"""


class Module(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Conv(Module):
    "Debugging in progress"
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        if type(kernel_size) is int: kernel_size = (kernel_size, kernel_size)
        self.kernel_size = tuple(kernel_size)
        self.dilation = int(dilation)
        self.padding = int(padding)
        self.stride = int(stride)

        self.input = None

        # Parameters initialization of torch.nn.Conv2d
        k = math.sqrt(1.0/(self.in_channels * reduce(lambda x, y: x * y, self.kernel_size)))
        self.weights = empty((self.out_channels, self.in_channels) + self.kernel_size).uniform_(-k, k)
        self.bias = empty((self.out_channels)).uniform_(-k, k)
        self.grad_weights = empty(self.weights.shape).fill_(0.0)
        self.grad_bias = empty(self.bias.shape).fill_(0.0)

    def forward(self, input):
        self.input = input
        output = unfold(input, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        output = self.weights.view((self.out_channels, -1)) @ output + self.bias.view((-1, 1))
        output = fold(output, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        return output

    def backward(self, gradwrtoutput):
        dL_dX = unfold(gradwrtoutput, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        dL_dX = self.weights.rot90().rot90().view((self.out_channels, -1)) @ dL_dX
        dL_dX = fold(dL_dX, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)

        dL_dW = unfold(gradwrtoutput, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        dL_dW = self.input.view((self.out_channels, -1)) @ dL_dW
        dL_dW = fold(dL_dW, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        self.grad_weights = dL_dW

        dL_db = gradwrtoutput.sum(dim=(0,1))
        self.grad_bias = dL_db

        return dL_dX

    def param(self):
        weights_pair = (self.weights, self.grad_weights)
        bias_pair = (self.bias, self.grad_bias)
        parameters = [weights_pair, bias_pair]
        return parameters


class ConvTranspose(Module):
    "Coding in progress"
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        if type(kernel_size) is int: kernel_size = (kernel_size, kernel_size)
        self.kernel_size = tuple(kernel_size)
        self.dilation = int(dilation)
        self.padding = int(padding)
        self.stride = int(stride)

        self.input = None

        # Parameters initialization of torch.nn.Conv2d
        k = math.sqrt(1.0/(self.in_channels * reduce(lambda x, y: x * y, self.kernel_size)))
        self.weights = empty((self.in_channels, self.out_channels) + self.kernel_size).uniform_(-k, k)
        self.bias = empty((self.out_channels)).uniform_(-k, k)
        self.grad_weights = empty(self.weights.shape).fill_(0.0)
        self.grad_bias = empty(self.bias.shape).fill_(0.0)


class Upsampling(Module):
    "Coding in progress"
    def __init__(self):
        pass


class ReLU(Module):
    def __init__(self):
        pass

    def forward(self, input):
        output = input.maximum(empty(input.shape).fill_(0.0))
        return output

    def backward(self, gradwrtoutput):
        grad = gradwrtoutput.gt(empty(gradwrtoutput.shape).fill_(0.0)).float()
        return grad


class Sigmoid(Module):
    def __init__(self):
        pass

    def forward(self, input):
        output = 1.0 / (1.0 + (-input).exp())
        return output

    def backward(self, gradwrtoutput):
        sigmoid = 1.0 / (1.0 + (-gradwrtoutput).exp())
        grad = sigmoid * (1.0 - sigmoid)
        return grad


class Sequential():
    def __init__(self, *modules):
        self.modules = [*modules]
        self.forward_initiated = False

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        output = input
        for mod in self.modules:
            output = mod.forward(output)
        self.forward_initiated = True
        return output

    def backward(self, output):
        gradwrtoutput = output
        if self.forward_initiated:
            for mod in reversed(self.modules):
                gradwrtoutput = mod.backward(gradwrtoutput)
        else:
            raise Exception("Forward pass must be performed before backward pass.")
        return gradwrtoutput

    def param(self):
        param = []
        for mod in self.modules:
            param += mod.param()
        return param


class MSELoss():
    "Accept 4-dimensional inputs N x C x W x H"
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, data1, data2):
        "data1 and data2 must have the same shape"
        assert data1.shape == data2.shape, "MSELoss called with two different-shaped inputs."
        loss_n = ((data1-data2)**2).sum(dim=tuple(range(1,data1.ndim)))
        Loss = 0.0
        if self.reduction == "sum":
            Loss = loss_n.sum()
        elif self.reduction == "mean":
            Loss = loss_n.mean()
        else:
            raise Exception("Loss reduction criterion is not defined.")
        return Loss


class SGD():
    """
    Parameters can be obtained from Sequential.param() function -> these parameters are a view (not a copy) of the real parameters
    lr is the learning rate
    """
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        "Set all grad Tensors to 0."
        for i in range(len(self.parameters)):
            self.parameters[i][1] = self.parameters[i][1].fill_(0.0)

    def step(self):
        for i in range(len(self.parameters)):
            old_param = self.parameters[i][0]
            grad = self.parameters[i][1]
            self.parameters[i][0] = old_param - self.lr * grad


if __name__ == '__main__':
    data_real = empty(10, 3, 32, 32).fill_(1.0)
    noise = empty(10, 3, 32, 32).normal_(mean=0.0, std=0.1)
    data_noised = data_real.clone().detach() + noise

    # Example of use (Conv does not work properly in this version, debugging in progress)
    Model = Sequential(ReLU())
    MSE = MSELoss()
    SGD_optimizer = SGD(Model.param(), lr=0.01)

    for i in range(100):
        predictions = Model(data_noised)
        loss = MSE(predictions, data_real)
        SGD_optimizer.zero_grad()
        Model.backward(loss)
        SGD_optimizer.step()

