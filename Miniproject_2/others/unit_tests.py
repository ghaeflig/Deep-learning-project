import time
import functools

# torch is only used here for testing purpose and not for the main part of miniproject2.
import torch
import torch.nn as nn
from torch import empty, cat, arange

import sys
sys.path.append("..")
import model as proj


class Test():
    """
    Testing class, all the tests run when the class is initialized.
    :param run_list: tests to run, defaults to None (all tests run)
    :type run_list: int (test index) or list-like of ints (test indices), optional
                    run_list = [] if the tests must not be run during instantiation
    """
    def __init__(self, run_list=None):
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # selects either GPU either CPU

        torch.manual_seed(0) # reproducibility

        self.tests = [] # tests done

        # create tests list (alphabetically)
        tests_list = []
        for object in dir(self):
            if object.startswith('test__'):
                tests_list.append(object)

        if run_list is None:
            run_list = list(range(1,len(tests_list)+1))
        elif type(run_list) is int:
            run_list = [run_list]
        else:
            run_list = list(run_list)

        # run and print tests
        for id, object in enumerate(tests_list):
            if id + 1 in run_list:
                test = getattr(self, object)
                time_beginning = time.time()
                test()
                time_end = time.time()
                runtime = str(round(time_end - time_beginning,10)).ljust(5,'0')[:5]  # runtime (5 characters long)
                self.tests[-1] = self.tests[-1] + [runtime]

                #  \033[91m  colors output in red
                #  \033[92m  colors output in green
                print(f"\033[{92 if self.tests[-1][0] else 91}m"
                      f"Test {id + 1} ({'passed' if self.tests[-1][0] else 'failed'}, {self.tests[-1][2]} sec): "
                      f"{self.tests[-1][1]}")

        # print final test result
        if len(self.tests) != 0:
            tests_passed = functools.reduce(lambda x, y: x + y[0], self.tests, 0)  # Number of tests passed
            tests_total = len(self.tests)  # Total number of tests
            tests_result = f"[{tests_passed}/{tests_total}]"
            if tests_passed == tests_total:
                print(f"\n\033[92m{tests_result} All tests passed")
            else:
                print(f"\n\033[91m{tests_result} Some tests failed")

        # reset output color
        print("\033[0m", end='')

    def print_tests(self):
        """ Prints the tests done and stored in self.tests. """
        for id, test in enumerate(self.tests):
            #  \033[91m  colors output in red
            #  \033[92m  colors output in green
            print(f"\033[{92 if test[0] else 91}m"
                  f"Test {id + 1} ({'passed' if test[0] else 'failed'}): "
                  f"{test[1]}")

        # print final test result
        tests_passed = functools.reduce(lambda x, y: x + y[0], self.tests, 0)  # Number of tests passed
        tests_total = len(self.tests)  # Total number of tests
        tests_result = f"[{tests_passed}/{tests_total}]"
        if tests_passed == tests_total:
            print(f"\n\033[92m{tests_result} All tests passed")
        else:
            print(f"\n\033[91m{tests_result} Some tests failed")

        # reset output color
        print("\033[0m", end='')

    def howclose(self, tensor1, tensor2, error_admitted=1e-4, print_result=True, return_indices=True):
        """
        Computes the number of successes and the number of failure for self.allclose.
        :param tensor1: first tensor
        :type tensor2: Tensor with the same shape as tensor2
        :param tensor2: second tensor
        :type tensor2: Tensor with the same shape as tensor1
        :param error_admitted: maximum error admitted between the two tensors, defaults to 1e-4
        :type error_admitted: float >= 0.0, optional
        :param print_result: print the result if True, defaults to True
        :type print_result: bool, optional
        :param return_indices: return the failure indices if True, defaults to True
        :type return_indices: bool, optional
        :return: 1) number of successes
                 2) total number of values
                 3) failure indices (only if return_indices True)
        :rtype: list [int, int(, indices)]
        """
        tensor_bool = (tensor1 - tensor2).abs().less_equal(error_admitted)
        result = [int(tensor_bool.sum()), tensor_bool.numel()]
        if print_result: print(f"{int(tensor_bool.sum())}/{tensor_bool.numel()}")
        if return_indices:
            indices = torch.nonzero(~tensor_bool, as_tuple=True)
            result = result + [indices]
        return result

    def allclose(self, tensor1, tensor2, error_admitted=1e-4):
        """
        Function used to compare the two tensor inputs and check if they are approximately equals.
        :param tensor1: first tensor
        :type tensor2: Tensor with the same shape as tensor2
        :param tensor2: second tensor
        :type tensor2: Tensor with the same shape as tensor1
        :param error_admitted: maximum error admitted between the two tensors, defaults to 1e-4
        :type error_admitted: float >= 0.0, optional
        :return: 1) True if all values of the tensors are close (less than error_admitted) else False
                 2) Score in string type with the number of successes
                 3) indices of the failures
        :rtype: list [bool, str, failure indices]
        """
        result = (tensor1 - tensor2).abs().less_equal(error_admitted).all()
        score = ""
        error_indices = None
        if result == False:
            howclose = self.howclose(tensor1=tensor1, tensor2=tensor2, error_admitted=error_admitted, print_result=False)
            score = f"{howclose[0]}/{howclose[1]}"
            error_indices = howclose[2]
        return [result, score, error_indices]

    def test__01(self):
        data = 2000 * torch.rand(size=(100, 50, 20, 10)).to(self.DEVICE) - 1000

        layer_torch = torch.nn.ReLU()
        result_torch = layer_torch(data)

        layer_test = proj.ReLU()
        result_test = layer_test(data)

        test = self.allclose(result_torch, result_test)
        score = ""
        if test[0] == False:
            score = f" [{test[1]}]"
        self.tests.append([test[0], f"ReLU forward pass" + score])

    def test__02(self):
        data = 2000 * torch.rand(size=(100, 50, 20, 10), requires_grad=True).to(self.DEVICE) - 1000
        data.retain_grad()

        layer_torch = torch.nn.ReLU()
        output_torch = layer_torch(data)
        output_torch.retain_grad()
        zero_tensor = empty(output_torch.shape).fill_(0.0).to(self.DEVICE)
        output_torch.dist(zero_tensor, p=2).backward()
        result_torch = data.grad

        layer_test = proj.ReLU()
        output_test = layer_test(data)
        result_test = layer_test.backward(output_torch.grad)

        test = self.allclose(result_torch, result_test)
        score = ""
        if test[0] == False:
            score = f" [{test[1]}]"
        self.tests.append([test[0], "ReLU forward/backward pass" + score])

    def test__03(self):
        data = 20 * torch.rand(size=(100, 50, 20, 10)).to(self.DEVICE) - 10

        layer_torch = torch.nn.Sigmoid()
        result_torch = layer_torch(data)

        layer_test = proj.Sigmoid()
        result_test = layer_test(data)

        test = self.allclose(result_torch, result_test)
        score = ""
        if test[0] == False:
            score = f" [{test[1]}]"
        self.tests.append([test[0], "Sigmoid forward pass" + score])

    def test__04(self):
        data = 20 * torch.rand(size=(100, 50, 20, 10), requires_grad=True).to(self.DEVICE) - 10
        data.retain_grad()

        layer_torch = torch.nn.Sigmoid()
        output_torch = layer_torch(data)
        output_torch.retain_grad()
        zero_tensor = empty(output_torch.shape).fill_(0.0).to(self.DEVICE)
        output_torch.dist(zero_tensor, p=2).backward()
        result_torch = data.grad

        layer_test = proj.Sigmoid()
        output_test = layer_test(data)
        result_test = layer_test.backward(output_torch.grad)

        test = self.allclose(result_torch, result_test)
        score = ""
        if test[0] == False:
            score = f" [{test[1]}]"
        self.tests.append([test[0], "Sigmoid forward/backward pass" + score])

    def test__05(self, in_channels=3, out_channels=5, kernel_size=3, dilation=2, padding=1, stride=2):
        data = 20 * torch.rand(size=(2, in_channels, 100, 100)).to(self.DEVICE) - 10

        layer_torch = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride).to(self.DEVICE)
        result_torch = layer_torch(data)

        layer_test = proj.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
        layer_test.weight = layer_torch.weight
        layer_test.bias = layer_torch.bias
        result_test = layer_test(data)

        test = self.allclose(result_torch, result_test)
        score = ""
        if test[0] == False:
            score = f" [{test[1]}]"
        score += f"\n\tin_channels={in_channels}, out_channels={out_channels}, kernel_size={out_channels}, dilation={dilation}, padding={padding}, stride={stride}"
        self.tests.append([test[0], "Conv2d forward pass" + score])

    def test__06(self, in_channels=3, out_channels=5, kernel_size=3, dilation=2, padding=1, stride=2):
        data = 20 * torch.rand(size=(2, in_channels, 100, 100), requires_grad=True).to(self.DEVICE) - 10
        data.retain_grad()

        layer_torch = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride).to(self.DEVICE)
        output_torch = layer_torch(data)
        output_torch.retain_grad()
        zero_tensor = empty(output_torch.shape).fill_(0.0).to(self.DEVICE)
        output_torch.dist(zero_tensor, p=2).backward()
        result_torch1 = data.grad
        result_torch2 = layer_torch.weight.grad
        result_torch3 = layer_torch.bias.grad

        layer_test = proj.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
        layer_test.weight = layer_torch.weight
        layer_test.bias = layer_torch.bias
        output_test = layer_test(data)
        result_test1 = layer_test.backward(output_torch.grad)
        result_test2 = layer_test.grad_weight
        result_test3 = layer_test.grad_bias

        test1 = self.allclose(result_torch1, result_test1)
        score = " (dL/dX"
        if test1[0] == False:
            score += f" [{test1[1]}]"

        test2 = self.allclose(result_torch2, result_test2, error_admitted=1e-3)
        score += ", dL/dW"
        if test2[0] == False:
            score += f" [{test2[1]}]"

        test3 = self.allclose(result_torch3, result_test3)
        score += ", dL/db"
        if test3[0] == False:
            score += f" [{test3[1]}]"

        test = test1[0] and test2[0] and test3[0]
        score += ")"
        score += f"\n\tin_channels={in_channels}, out_channels={out_channels}, kernel_size={out_channels}, dilation={dilation}, padding={padding}, stride={stride}"
        self.tests.append([test, "Conv2d forward/backward pass" + score])

    def test__07(self, in_channels=3, out_channels=5, kernel_size=3, dilation=2, padding=1, stride=2, output_padding=1):
        data = 20 * torch.rand(size=(2, in_channels, 49, 49)).to(self.DEVICE) - 10

        layer_torch = torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride, output_padding=output_padding).to(self.DEVICE)
        result_torch = layer_torch(data)

        layer_test = proj.TransposeConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride, output_padding=output_padding)
        layer_test.weight = layer_torch.weight
        layer_test.bias = layer_torch.bias
        result_test = layer_test(data)

        test = self.allclose(result_torch, result_test)
        score = ""
        if test[0] == False:
            score = f" [{test[1]}]"
        score += f"\n\tin_channels={in_channels}, out_channels={out_channels}, kernel_size={out_channels}, dilation={dilation}, padding={padding}, stride={stride}, output_padding={output_padding}"
        self.tests.append([test[0], "TransposeConv2d forward pass" + score])

    def test__08(self, in_channels=3, out_channels=5, kernel_size=3, dilation=2, padding=1, stride=2, output_padding=1):
        data = 20 * torch.rand(size=(2, in_channels, 100, 100), requires_grad=True).to(self.DEVICE) - 10
        data.retain_grad()

        layer_torch = torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride, output_padding=output_padding).to(self.DEVICE)
        output_torch = layer_torch(data)
        output_torch.retain_grad()
        zero_tensor = empty(output_torch.shape).fill_(0.0).to(self.DEVICE)
        output_torch.dist(zero_tensor, p=2).backward()
        result_torch1 = data.grad
        result_torch2 = layer_torch.weight.grad
        result_torch3 = layer_torch.bias.grad

        layer_test = proj.TransposeConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride, output_padding=output_padding)
        layer_test.weight = layer_torch.weight
        layer_test.bias = layer_torch.bias
        output_test = layer_test(data)
        result_test1 = layer_test.backward(output_torch.grad)
        result_test2 = layer_test.grad_weight
        result_test3 = layer_test.grad_bias


        test1 = self.allclose(result_torch1, result_test1)
        score = " (dL/dX"
        if test1[0] == False:
            score += f" [{test1[1]}]"

        test2 = self.allclose(result_torch2, result_test2)
        score += ", dL/dW"
        if test2[0] == False:
            score += f" [{test2[1]}]"

        test3 = self.allclose(result_torch3, result_test3)
        score += ", dL/db"
        if test3[0] == False:
            score += f" [{test3[1]}]"

        test = test1[0] and test2[0] and test3[0]
        score += ")"
        score += f"\n\tin_channels={in_channels}, out_channels={out_channels}, kernel_size={out_channels}, dilation={dilation}, padding={padding}, stride={stride}, output_padding={output_padding}"
        self.tests.append([test, "TransposeConv2d forward/backward pass" + score])

    def test__09(self):
        data = 2 * torch.rand(size=((100, 50, 20, 10))).to(self.DEVICE) - 1
        target = empty(data.shape).fill_(0.0).to(self.DEVICE)

        layer_torch1 = torch.nn.MSELoss(reduction="mean")
        result_torch1 = layer_torch1(data, target)

        layer_test1 = proj.MSE(reduction="mean")
        result_test1 = layer_test1(data, target)

        layer_torch2 = torch.nn.MSELoss(reduction="sum")
        result_torch2 = layer_torch2(data, target)

        layer_test2 = proj.MSE(reduction="sum")
        result_test2 = layer_test2(data, target)

        test1 = self.allclose(result_torch1, result_test1)
        score = " (mean"
        if test1[0] == False:
            score += f" [{test1[1]}]"

        test2 = self.allclose(result_torch2, result_test2)
        score += ", sum"
        if test2[0] == False:
            score += f" [{test2[1]}]"

        test = test1[0] and test2[0]
        score += ")"
        self.tests.append([test, "MSELoss forward pass" + score])

    def test__10(self):
        data1 = 2 * torch.rand(size=((100, 50, 20, 10)), requires_grad=True).to(self.DEVICE) - 1
        data1.retain_grad()
        target1 = empty(data1.shape).fill_(0.0).to(self.DEVICE)

        layer_torch1 = torch.nn.MSELoss(reduction="mean")
        output_torch1 = layer_torch1(data1, target1)
        output_torch1.retain_grad()
        zero_tensor1 = empty(output_torch1.shape).fill_(0.0).to(self.DEVICE)
        output_torch1.dist(zero_tensor1, p=2).backward()
        result_torch1 = data1.grad

        layer_test1 = proj.MSE(reduction="mean")
        output_test1 = layer_test1(data1, target1)
        result_test1 = layer_test1.backward(output_torch1.grad)

        data2 = 2 * torch.rand(size=((100, 50, 20, 10)), requires_grad=True).to(self.DEVICE) - 1
        data2.retain_grad()
        target2 = empty(data1.shape).fill_(0.0).to(self.DEVICE)

        layer_torch2 = torch.nn.MSELoss(reduction="sum")
        output_torch2 = layer_torch2(data2, target2)
        output_torch2.retain_grad()
        zero_tensor2 = empty(output_torch2.shape).fill_(0.0).to(self.DEVICE)
        output_torch2.dist(zero_tensor2, p=2).backward()
        result_torch2 = data2.grad

        layer_test2 = proj.MSE(reduction="sum")
        output_test2 = layer_test2(data2, target2)
        result_test2 = layer_test2.backward(output_torch2.grad)

        test1 = self.allclose(result_torch1, result_test1)
        score = " (mean"
        if test1[0] == False:
            score += f" [{test1[1]}]"

        test2 = self.allclose(result_torch2, result_test2)
        score += ", sum"
        if test2[0] == False:
            score += f" [{test2[1]}]"

        test = test1[0] and test2[0]
        score += ")"
        self.tests.append([test, "MSELoss forward/backward pass" + score])

    def test__11(self):
        data = 20 * torch.rand(size=(2, 3, 100, 100), requires_grad=True).to(self.DEVICE) - 10
        data.retain_grad()

        layer_torch = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, dilation=1, padding=1, stride=2).to(self.DEVICE)
        output_torch = layer_torch(data)
        output_torch.retain_grad()
        zero_tensor = empty(output_torch.shape).fill_(0.0).to(self.DEVICE)
        output_torch.dist(zero_tensor, p=2).backward()
        result_torch1 = data.grad
        output_torch_grad = output_torch.grad.clone().detach()
        weight = layer_torch.weight.clone().detach()
        bias = layer_torch.bias.clone().detach()
        SGD_torch = torch.optim.SGD(layer_torch.parameters(), lr=10, weight_decay=0.1)
        SGD_torch.step()
        result_torch2 = layer_torch.weight
        result_torch3 = layer_torch.bias

        layer_test = proj.Conv2d(in_channels=3, out_channels=6, kernel_size=3, dilation=1, padding=1, stride=2)
        layer_test.weight = weight
        layer_test.bias = bias
        output_test = layer_test(data)
        result_test1 = layer_test.backward(output_torch_grad)
        SGD_test = proj.SGD(layer_test.param(), lr=10, weight_decay=0.1)
        SGD_test.step()
        result_test2 = layer_test.weight
        result_test3 = layer_test.bias

        test1 = self.allclose(result_torch1, result_test1)
        score = " (dL/dX"
        if test1[0] == False:
            score += f" [{test1[1]}]"

        test2 = self.allclose(result_torch2, result_test2, error_admitted=1e-3)
        score += ", W update"
        if test2[0] == False:
            score += f" [{test2[1]}]"

        test3 = self.allclose(result_torch3, result_test3)
        score += ", b update"
        if test3[0] == False:
            score += f" [{test3[1]}]"

        # print(result_torch3[test3[2]]) # debugging
        # print(result_test3[test3[2]]) # debugging

        test = test1[0] and test2[0] and test3[0]
        score += ")"
        self.tests.append([test, "Conv2d forward/backward pass with SGD" + score])

    def test__12(self):
        data = 20 * torch.rand(size=(2, 3, 100, 100), requires_grad=True).to(self.DEVICE) - 10

        model_torch = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, dilation=1, padding=1, stride=2),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, dilation=1, padding=1, stride=2),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(in_channels=12, out_channels=6, kernel_size=3, dilation=1, padding=1, stride=2, output_padding=1),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=3, dilation=1, padding=1, stride=2, output_padding=1),
                                    nn.Sigmoid()).to(self.DEVICE)
        weights = []
        biases = []
        for module in model_torch:
            if 'weight' in dir(module):
                weights.append(module.weight.clone().detach())
            else:
                weights.append(None)
            if 'bias' in dir(module):
                biases.append(module.bias.clone().detach())
            else:
                biases.append(None)
        SGD_torch = torch.optim.SGD(model_torch.parameters(), lr=10, weight_decay=0.0)
        MSE_torch = nn.MSELoss()
        data_torch = data.clone()
        data_torch.retain_grad()
        predictions_torch = model_torch(data_torch)
        SGD_torch.zero_grad()
        loss_torch = MSE_torch.forward(predictions_torch, data_torch)
        loss_torch.retain_grad()
        loss_torch.backward()
        SGD_torch.step()

        model_test = proj.Sequential(proj.Conv2d(in_channels=3, out_channels=6, kernel_size=3, dilation=1, padding=1, stride=2),
                                     proj.ReLU(),
                                     proj.Conv2d(in_channels=6, out_channels=12, kernel_size=3, dilation=1, padding=1, stride=2),
                                     proj.ReLU(),
                                     proj.TransposeConv2d(in_channels=12, out_channels=6, kernel_size=3, dilation=1, padding=1, stride=2, output_padding=1),
                                     proj.ReLU(),
                                     proj.TransposeConv2d(in_channels=6, out_channels=3, kernel_size=3, dilation=1, padding=1, stride=2, output_padding=1),
                                     proj.Sigmoid())
        for id, module in enumerate(model_test):
            if weights[id] != None:
                module.weight = weights[id]
            if biases[id] != None:
                module.bias = biases[id]
        SGD_test = proj.SGD(model_test.param(), lr=10, weight_decay=0.0)
        MSE_test = proj.MSE()
        data_test = data.clone().detach()
        predictions_test = model_test(data_test)
        SGD_test.zero_grad()
        loss_test = MSE_test.forward(predictions_test, data_test)
        grad_loss_test = MSE_test.backward(empty(1).fill_(1.0).to(self.DEVICE))
        model_test.backward(grad_loss_test)
        SGD_test.step()

        global_test = True
        score = ""
        for id, module in enumerate(model_torch):
            if 'weight' in dir(module):
                test = self.allclose(model_torch[id].weight, model_test[id].weight)
                score += ", layer1.W"
                if test[0] == False:
                    global_test = False
                    score += f" [{test[1]}]"
            if 'bias' in dir(module):
                test = self.allclose(model_torch[id].bias, model_test[id].bias)
                score += ", layer1.b"
                if test[0] == False:
                    global_test = False
                    score += f" [{test[1]}]"
        score = " (" + score[2:] + ")"
        self.tests.append([global_test, "Parameter updates for Model forward/backward pass with SGD" + score])


if __name__ == '__main__':
    torch.set_printoptions(precision=8)
    Test() # runs all tests after being instantiated
    print("\n\n")

    # Model creation example and testing of each of its Modules
    model = proj.Sequential(proj.Conv2d(in_channels=3, out_channels=12, kernel_size=3, dilation=1, padding=1, stride=2),
                            proj.ReLU(),
                            proj.Conv2d(in_channels=12, out_channels=48, kernel_size=3, dilation=1, padding=1, stride=2),
                            proj.ReLU(),
                            proj.TransposeConv2d(in_channels=48, out_channels=12, kernel_size=3, dilation=1, padding=1, stride=2, output_padding=1),
                            proj.ReLU(),
                            proj.TransposeConv2d(in_channels=12, out_channels=3, kernel_size=3, dilation=1, padding=1, stride=2, output_padding=1),
                            proj.Sigmoid())
    TestModel = Test(run_list=[]) # no tests run during instantiation
    for module in model.modules:
        module.test(TestModel)
    TestModel.print_tests()

