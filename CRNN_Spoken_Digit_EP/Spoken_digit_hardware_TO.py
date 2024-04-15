from __future__ import print_function
import argparse
import os
import math
import numpy as np
import copy
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

from torch import _VF
from torch.nn.utils.rnn import PackedSequence
from torch.utils.data import TensorDataset, DataLoader

# Set CUDA device to be used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Global variables for script configuration
args = None
limit_size_start = None
limit_size_stop = None
limit_size_step = None
limit_size = None


# Load pre-existing hardware weights
hardware_weight = torch.load('CRNN_RRAM_hardware_weight.pt', map_location='cpu')
weight_list_one = []
weight_list_zero = []

# Model structure detailing the dimensions of each weight tensor
model_state_dict = {
    'conv1_weight': [64, 1, 3, 2],
    'conv2_weight': [32, 64, 3, 2],
    'rnn_ih_weight': [128, 32],
    'rnn_hh_weight': [128, 128],
    'fc1_weight': [256, 128],
    'fc2_weight': [10, 256],
}

index = 0
# Determine midpoint of the 'set' weight array for logical separation
half_size = hardware_weight["set"].shape[-1] // 2
for i in model_state_dict.keys():
    if "weight" in i:
        size = np.prod(model_state_dict[i])  # Total number of elements in the weight tensor
        # Extract and reshape the 'set' and 'reset' weights for each layer, both original and noise-injected
        weight_list_one.append((hardware_weight["set"][:, index: index + size] - hardware_weight["set"][:,
                                                                                 half_size + index: half_size + index + size]).reshape(
            [hardware_weight["set"].shape[0]] + model_state_dict[i]))
        weight_list_zero.append((hardware_weight["reset"][:, index: index + size] - hardware_weight["set"][:,
                                                                                    half_size + index: half_size + index + size]).reshape(
            [hardware_weight["set"].shape[0]] + model_state_dict[i]))

        index += size

# Clear the lists to free up memory as they are no longer needed after processing
hardware_weight = []
hardware_weight_n = []


def load_from_npz(file_dir):
    """
    Loads data from a .npz file and aggregates it into a list.

    Args:
        file_dir: The directory path of the .npz file.

    Returns:
        A numpy array containing all the items loaded from the .npz file.
    """
    a = np.load(file_dir, allow_pickle=True)  # Load the .npz file
    r_list = []  # Initialize an empty list to hold the data
    for j in a:  # Iterate through each item in the loaded .npz file
        for i in a[j]:  # Iterate through each element in the item
            r_list.append(i)  # Append the element to the list
    return np.array(r_list)  # Return the list as a numpy array


def make_dataset(dataset):
    """
    Converts a dataset of features and labels into PyTorch TensorDataset.

    Args:
        dataset: A list of dictionaries, each containing 'feature' and 'label'.

    Returns:
        A PyTorch TensorDataset containing the features and labels as tensors.
    """
    feature_list = []  # List to store feature tensors
    label_list = []  # List to store label tensors
    for i in dataset:  # Iterate through the dataset
        # Reshape the 'feature' and 'label' and append them to their respective lists
        feature_list.append(np.reshape(i['feature'], newshape=[23, 20]))
        label_list.append(np.reshape(i['label'], newshape=1))
    # Convert lists to numpy arrays, then to PyTorch tensors, and finally to TensorDataset
    dataset = TensorDataset((torch.from_numpy(np.array(feature_list)).float()),
                            (torch.from_numpy(np.array(label_list)).long()))

    return dataset


def quantization(m, bit, after_relu=False):
    """
    Quantizes a tensor to a specified bit width.

    Args:
        m: The tensor to quantize.
        bit: The bit width to quantize the tensor to.
        after_relu: A boolean indicating whether the quantization is applied after a ReLU activation.

    Returns:
        The quantized tensor.
    """
    out = m.clone()  # Clone the input tensor to avoid modifying it directly
    scale = (out.max() - out.min()) / (2 ** bit - 1)  # Calculate the scale for quantization
    if after_relu:
        # Quantize the tensor for post-ReLU data
        out = torch.quantize_per_tensor(out, scale=scale, zero_point=0, dtype=torch.quint8)
        # Dequantize back to float for further processing
        out = (out.int_repr() * scale).float().cuda()
    else:
        # Calculate zero points for general case quantization
        zp_max = out.max() // scale
        zp_min = out.min() // scale
        zp = zp_min + 2 ** (bit - 1) - zp_max - 1
        # Quantize the tensor
        out = torch.quantize_per_tensor((out - out.min()), scale=scale, zero_point=zp, dtype=torch.qint32)
        # Dequantize back to float for further processing
        out = ((out.int_repr() - zp + zp_min) * scale).float().cuda()

    return out


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k, mask_old, score_old):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        # limit
        if mask_old.sum() != mask_old.numel():
            d_s = (score_old - scores.clone()).abs() / score_old
            out = torch.where(d_s < limit_size, mask_old, out)
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None, None, None


class MaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        for i in range(len(weight_list_one)):
            if weight_list_one[i][0].shape == self.weight.data.shape:
                self.hardware_weight_one = torch.from_numpy(weight_list_one[i].copy()).abs().float().cuda()
                self.hardware_weight_zero = torch.from_numpy(weight_list_zero[i].copy()).abs().float().cuda()
                break
            else:
                pass

        # NOTE: initialize the weights like this.
        fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
        # scale:
        fan = fan * (1 - 0.5)
        gain = nn.init.calculate_gain("relu")
        self.std = gain / math.sqrt(fan)
        self.scale = self.hardware_weight_one.mean() / self.std
        self.sign = self.weight.data.sign().cuda()
        self.subnet_old = torch.ones_like(self.weight.data).cuda()
        self.score_old = self.scores.data.clone()
        self.weight.requires_grad = False

    def forward(self, x, count, switch_count=None):
        subnet = GetSubnet.apply(self.scores.abs(), args.sparsity, self.subnet_old, self.score_old.abs())
        if switch_count is not None:
            switch_count[-1] += torch.where(subnet != switch_count[0], 1.0, 0.0).cuda()
            self.subnet_old = subnet.data
            self.score_old = self.scores.data.clone()
            w = self.sign * (self.hardware_weight_one[count] * subnet + self.hardware_weight_zero[count] * (1 - subnet))
            x = F.conv2d(
                x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
            switch_count[0] = subnet.data.clone()
            return x / self.scale, switch_count
        else:
            w = self.sign * (self.hardware_weight_one[count] * subnet + self.hardware_weight_zero[count] * (1 - subnet))
            x = F.conv2d(
                x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
            return x / self.scale


class MaskRNN(nn.RNNBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialization of scores for input-hidden and hidden-hidden weights.
        # Initialization of actual weights using a specific scheme.
        # Matching and loading hardware-simulated weights for further use in forward pass.

        self.ih_scores = nn.Parameter(torch.Tensor(self.weight_ih_l0.size()))
        nn.init.kaiming_uniform_(self.ih_scores, a=math.sqrt(5))

        self.hh_scores = nn.Parameter(torch.Tensor(self.weight_hh_l0.size()))
        nn.init.kaiming_uniform_(self.hh_scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        ih_fan = nn.init._calculate_correct_fan(self.weight_ih_l0, "fan_in")
        # scale:
        ih_fan = ih_fan * (1 - 0.5)
        ih_gain = nn.init.calculate_gain("relu")
        self.ih_std = ih_gain / math.sqrt(ih_fan)
        self.weight_ih_l0.data = self.weight_ih_l0.data.sign() * self.ih_std

        hh_fan = nn.init._calculate_correct_fan(self.weight_hh_l0, "fan_in")
        # scale:
        hh_fan = hh_fan * (1 - 0.5)
        hh_gain = nn.init.calculate_gain("relu")
        self.hh_std = hh_gain / math.sqrt(hh_fan)
        self.weight_hh_l0.data = self.weight_hh_l0.data.sign() * self.hh_std

        # NOTE: turn the gradient on the weights off
        self.weight_ih_l0.requires_grad = False
        self.weight_hh_l0.requires_grad = False

        for i in range(len(weight_list_one)):
            if weight_list_one[i][0].shape == self.weight_ih_l0.data.shape:
                self.hardware_weight_one_ih = torch.from_numpy(weight_list_one[i].copy()).abs().float().cuda()
                self.hardware_weight_zero_ih = torch.from_numpy(weight_list_zero[i].copy()).abs().float().cuda()

            elif weight_list_one[i][0].shape == self.weight_hh_l0.data.shape:
                self.hardware_weight_one_hh = torch.from_numpy(weight_list_one[i].copy()).abs().float().cuda()
                self.hardware_weight_zero_hh = torch.from_numpy(weight_list_zero[i].copy()).abs().float().cuda()
            else:
                pass

        self.ih_subnet_old = torch.ones_like(self.weight_ih_l0.data).cuda()
        self.ih_score_old = self.ih_scores.data.clone()
        self.ih_scale = self.hardware_weight_one_ih.mean() / self.ih_std
        self.ih_sign = self.weight_ih_l0.data.sign().cuda()

        self.hh_subnet_old = torch.ones_like(self.weight_hh_l0.data).cuda()
        self.hh_score_old = self.hh_scores.data.clone()
        self.hh_scale = self.hardware_weight_one_hh.mean() / self.hh_std
        self.hh_sign = self.weight_hh_l0.data.sign().cuda()

    def forward(self, input, count, hx=None, switch_count_ih=None, switch_count_hh=None):
        # Handling of both packed and unpacked input sequences.
        # Creation of supermasks for both sets of weights.
        # Selection of hardware-simulated weights based on the current supermasks and application of noise if indicated by `switch_count`.
        # Execution of the RNN step using either tanh or relu activation, as specified by the `mode` attribute.
        orig_input = input
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
                if hx is not None:
                    if hx.dim() != 2:
                        raise RuntimeError(
                            f"For unbatched 2-D input, hx should also be 2-D but got {hx.dim()}-D tensor")
                    hx = hx.unsqueeze(1)
            else:
                if hx is not None and hx.dim() != 3:
                    raise RuntimeError(
                        f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor")
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.hidden_size,
                             dtype=input.dtype, device=input.device)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        assert hx is not None
        self.check_forward_args(input, hx, batch_sizes)
        assert self.mode == 'RNN_TANH' or self.mode == 'RNN_RELU'

        weight_ih_l0_subnet = GetSubnet.apply(self.ih_scores.abs(), args.sparsity, self.ih_subnet_old,
                                              self.ih_score_old.abs())
        weight_hh_l0_subnet = GetSubnet.apply(self.hh_scores.abs(), args.sparsity, self.hh_subnet_old,
                                              self.hh_score_old.abs())

        if switch_count_ih is not None:
            self.ih_subnet_old = weight_ih_l0_subnet.data
            self.ih_score_old = self.ih_scores.data.clone()
            self.hh_subnet_old = weight_hh_l0_subnet.data
            self.hh_score_old = self.hh_scores.data.clone()

            switch_count_ih[-1] += torch.where(weight_ih_l0_subnet != switch_count_ih[0], 1.0, 0.0).cuda()
            switch_count_hh[-1] += torch.where(weight_hh_l0_subnet != switch_count_hh[0], 1.0, 0.0).cuda()

            w_l = [
                self.ih_sign * (self.hardware_weight_one_ih[count] * weight_ih_l0_subnet + self.hardware_weight_zero_ih[
                    count] * (1 - weight_ih_l0_subnet)) / self.ih_scale,

                self.hh_sign * (self.hardware_weight_one_hh[count] * weight_hh_l0_subnet + self.hardware_weight_zero_hh[
                    count] * (1 - weight_hh_l0_subnet)) / self.hh_scale
            ]
            switch_count_ih[0] = weight_ih_l0_subnet.data.clone()
            switch_count_hh[0] = weight_hh_l0_subnet.data.clone()
        else:
            w_l = [
                self.ih_sign * (self.hardware_weight_one_ih[count] * weight_ih_l0_subnet + self.hardware_weight_zero_ih[
                    count] * (1 - weight_ih_l0_subnet)) / self.ih_scale,

                self.hh_sign * (self.hardware_weight_one_hh[count] * weight_hh_l0_subnet + self.hardware_weight_zero_hh[
                    count] * (1 - weight_hh_l0_subnet)) / self.hh_scale
            ]

        if batch_sizes is None:
            if self.mode == 'RNN_TANH':
                result = _VF.rnn_tanh(input, hx, w_l, self.bias, self.num_layers,
                                      self.dropout, self.training, self.bidirectional,
                                      self.batch_first)
            else:
                result = _VF.rnn_relu(input, hx, w_l, self.bias, self.num_layers,
                                      self.dropout, self.training, self.bidirectional,
                                      self.batch_first)
        else:
            if self.mode == 'RNN_TANH':
                result = _VF.rnn_tanh(input, batch_sizes, hx, w_l, self.bias,
                                      self.num_layers, self.dropout, self.training,
                                      self.bidirectional)
            else:
                result = _VF.rnn_relu(input, batch_sizes, hx, w_l, self.bias,
                                      self.num_layers, self.dropout, self.training,
                                      self.bidirectional)

        output = result[0]
        hidden = result[1]

        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)

        if not is_batched:
            output = output.squeeze(batch_dim)
            hidden = hidden.squeeze(1)

        # return torch.mean(output, dim=1)
        if switch_count_ih is not None:
            return output, switch_count_ih, switch_count_hh
        else:
            return output


class MaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        for i in range(len(weight_list_one)):
            if weight_list_one[i][0].shape == self.weight.data.shape:
                self.hardware_weight_one = torch.from_numpy(weight_list_one[i].copy()).abs().float().cuda()
                self.hardware_weight_zero = torch.from_numpy(weight_list_zero[i].copy()).abs().float().cuda()
                break
            else:
                pass

        # NOTE: initialize the weights like this.
        fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
        # scale:
        fan = fan * (1 - 0.5)
        gain = nn.init.calculate_gain("relu")
        self.std = gain / math.sqrt(fan)
        self.scale = self.hardware_weight_one.mean() / self.std
        self.sign = self.weight.data.sign().cuda()
        self.subnet_old = torch.ones_like(self.weight.data).cuda()
        self.score_old = self.scores.data.clone()
        self.weight.requires_grad = False

    def forward(self, x, count, switch_count=None):
        subnet = GetSubnet.apply(self.scores.abs(), args.sparsity, self.subnet_old, self.score_old.abs())
        if switch_count is not None:
            switch_count[-1] += torch.where(subnet != switch_count[0], 1.0, 0.0).cuda()
            self.subnet_old = subnet.data
            self.score_old = self.scores.data.clone()
            w = self.sign * (self.hardware_weight_one[count] * subnet + self.hardware_weight_zero[count] * (1 - subnet))
            switch_count[0] = subnet.data.clone()
            return F.linear(x, w, self.bias) / self.scale, switch_count
        else:
            w = self.sign * (self.hardware_weight_one[count] * subnet + self.hardware_weight_zero[count] * (1 - subnet))
            return F.linear(x, w, self.bias) / self.scale


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = MaskConv(model_state_dict['conv1_weight'][1], model_state_dict['conv1_weight'][0],
                                   model_state_dict['conv1_weight'][-2:], [2, 1], bias=False)
        self.c_bn1 = nn.BatchNorm2d(model_state_dict['conv1_weight'][0], affine=False)
        self.conv2 = MaskConv(model_state_dict['conv2_weight'][1], model_state_dict['conv2_weight'][0],
                                   model_state_dict['conv2_weight'][-2:], [2, 1], bias=False)
        self.c_bn2 = nn.BatchNorm2d(model_state_dict['conv2_weight'][0], affine=False)
        self.rnn = MaskRNN(mode='RNN_RELU', input_size=model_state_dict['rnn_ih_weight'][-1],
                                hidden_size=model_state_dict['rnn_ih_weight'][0], num_layers=1, bias=False,
                                batch_first=False)
        self.fc1 = MaskLinear(model_state_dict['fc1_weight'][-1], model_state_dict['fc1_weight'][0], bias=False)
        self.fc2 = MaskLinear(model_state_dict['fc2_weight'][-1], model_state_dict['fc2_weight'][0], bias=False)

    def forward(self, x, qb=0, count=0, switch_count_list=None):
        x = x.unsqueeze(dim=1)
        x.data = quantization(x.data, qb)
        if switch_count_list:
            count = 0
            x, switch_count_list[0] = self.conv1(x, count, switch_count_list[0])
            x = F.max_pool2d(x, 2)
            x = self.c_bn1(x)
            x = F.relu(x)
            x.data = quantization(x.data, qb, after_relu=True)
            x, switch_count_list[1] = self.conv2(x, count, switch_count_list[1])
            x = F.max_pool2d(x, 2)
            x = self.c_bn2(x)
            x = F.relu(x)
            x = x.squeeze(dim=2)
            x.data = quantization(x.data, qb, after_relu=True)
            x = torch.transpose(x, -2, -1)
            x = torch.transpose(x, 0, 1)
            h_t = torch.zeros(self.rnn.num_layers, x.shape[1], self.rnn.hidden_size, dtype=x.dtype, device=x.device)
            for l in range(x.shape[0]):
                h_t, switch_count_list[2], switch_count_list[3] = self.rnn(x[l].unsqueeze(dim=0), count, h_t,
                                                                           switch_count_list[2], switch_count_list[3])
                if l:
                    h_all = torch.cat((h_all, h_t), dim=0)
                else:
                    h_all = h_t
                if l != x.shape[0] - 1:
                    h_t.data = quantization(h_t.data, qb, after_relu=True)
            aver_h = torch.mean(h_all, dim=0).reshape([x.shape[1], self.rnn.hidden_size])
            x = aver_h
            x.data = quantization(x.data, qb)
            x, switch_count_list[4] = self.fc1(x, count, switch_count_list[4])
            x = F.relu(x)
            x.data = quantization(x.data, qb, after_relu=True)
            x, switch_count_list[5] = self.fc2(x, count, switch_count_list[5])
            output = F.log_softmax(x, dim=1)
            count += 1
            return output, count, switch_count_list
        else:
            x = self.conv1(x, count)
            x = F.max_pool2d(x, 2)
            x = self.c_bn1(x)
            x = F.relu(x)
            x.data = quantization(x.data, qb, after_relu=True)
            x = self.conv2(x, count)
            x = F.max_pool2d(x, 2)
            x = self.c_bn2(x)
            x = F.relu(x)
            x = x.squeeze(dim=2)
            x.data = quantization(x.data, qb, after_relu=True)
            x = torch.transpose(x, -2, -1)
            x = torch.transpose(x, 0, 1)
            h_t = torch.zeros(self.rnn.num_layers, x.shape[1], self.rnn.hidden_size, dtype=x.dtype, device=x.device)
            for l in range(x.shape[0]):
                h_t = self.rnn(x[l].unsqueeze(dim=0), count, h_t)
                if l:
                    h_all = torch.cat((h_all, h_t), dim=0)
                else:
                    h_all = h_t
                if l != x.shape[0] - 1:
                    h_t.data = quantization(h_t.data, qb, after_relu=True)
            aver_h = torch.mean(h_all, dim=0).reshape([x.shape[1], self.rnn.hidden_size])
            x = aver_h
            x.data = quantization(x.data, qb)
            x = self.fc1(x, count)
            x = F.relu(x)
            x.data = quantization(x.data, qb, after_relu=True)
            x = self.fc2(x, count)
            output = F.log_softmax(x, dim=1)
            count += 1
            return output, count


def train(model, qb, switch_count_list, device, train_loader, optimizer, criterion, epoch):
    model.train()
    count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.squeeze(dim=1).to(device)
        optimizer.zero_grad()
        output, count, switch_count_list = model(data, qb=qb, count=count, switch_count_list=switch_count_list)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLimit sizes: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), limit_size))
    return switch_count_list


def Test(model, qb, device, test_loader):
    model.eval()
    correct = 0
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.squeeze(dim=1).to(device)
            output, count = model(data, qb=qb, count=count)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        print('\nHardware read times: {}, Test set: Accuracy: {}/{} ({:.2f}%)\n'.format(
            count, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return correct


def main():
    global args
    global limit_size_start
    global limit_size_stop
    global limit_size_step
    global limit_size

    params = {
        'epochs': 60,
        'q_bit': 3,
        'lr': 0.1,
        'momentum': 0.9,
        'wd': 0.0005,
        'batch_size': 30,
        'limit_size_start': 0.009,
        'limit_size_stop': 0.007,
        'limit_size_step': 10,
        'sparsity': 0.5,
    }

    # Training settings
    parser = argparse.ArgumentParser(description="Pytorch CRNN with Hardware TO")

    parser.add_argument('--batch-size', type=int, default=30, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=9, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--q_bit', type=int, default=3, metavar='N',
                        help='bit of quantized input (default: 3)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=0.0005, metavar='M',
                        help='Weight decay (default: 0.0005)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--data', type=str, default='data', help='Location to store data')
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='how sparse is each layer')
    args = parser.parse_args()
    args.sparsity = params['sparsity']
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    limit_size_start = params['limit_size_start']
    limit_size_stop = params['limit_size_stop']
    limit_size_step = params['limit_size_step']
    limit_size = limit_size_start

    train_dataset = load_from_npz("train_features.npz")
    test_dataset = load_from_npz("test_features.npz")

    train_loader = DataLoader(make_dataset(train_dataset),
                              batch_size=params['batch_size'], drop_last=True, shuffle=True, **kwargs)
    test_loader = DataLoader(make_dataset(test_dataset),
                             batch_size=args.test_batch_size, drop_last=True, shuffle=False, **kwargs)

    model = Net().to(device)
    # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important!
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=params['lr'],
        # momentum=params['momentum'],
        # weight_decay=params['wd'],
    )
    # scheduler = CosineAnnealingLR(optimizer, T_max=params['epochs'])
    criterion = nn.CrossEntropyLoss().to(device)
    best = 0
    switch_count_list = []
    for sl in [p for p in model.parameters() if p.requires_grad]:
        switch_count_list.append([
            torch.ones_like(sl.data).cuda(),
            torch.zeros_like(sl.data).cuda()
        ])

    acc_list = []

    for epoch in range(1, params['epochs'] + 1):
        switch_count_list = train(model, params['q_bit'], switch_count_list, device, train_loader, optimizer, criterion,
                                  epoch)
        correct = Test(model, params['q_bit'], device, test_loader)
        acc_list.append([correct, np.sum([i[1].cpu().sum() for i in switch_count_list])])
        if best <= correct:
            best = correct
            best_dict = copy.deepcopy(model.state_dict())
            limit_size = limit_size - (limit_size_start - limit_size_stop) / limit_size_step
            best_switch_count_list = copy.deepcopy(switch_count_list)
            print("best accuracy!\n")

        # scheduler.step()

    print("best accuracy: {:.2f}%\n".format(100. * best / len(test_loader.dataset)))

    # Obtain the Current Time
    now = datetime.datetime.now()
    # Convert the Timestamp to a String
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")

    if args.save_model:
        torch.save(best_dict, "spoken_digit_hardware_ep_program_{}_".format(best) + timestamp_str + ".pt")
        shape_list = []
        for i in best_dict.keys():
            if "weight" in i:
                shape_list.append([k for k in best_dict[i].shape])
        with open("spoken_digit_hardware_ep_program_{}_".format(best) + timestamp_str + ".txt", 'w') as f:
            f.write('arch:\n')
            for s in shape_list:
                arch = str(s[0])
                for t in s[1:]:
                    arch = arch + '_{}'.format(t)
                f.write('\t' + arch + '\n')
            f.write('training_epochs: {}\n'.format(params['epochs']))
            f.write('batch_size: {}\n'.format(params['batch_size']))
            f.write('input_bit: {}\n'.format(params['q_bit']))
            f.write('limit_size_start: {}\n'.format(params['limit_size_start']))
            f.write('limit_size_stop: {}\n'.format(params['limit_size_stop']))
            f.write('limit_size_step: {}\n'.format(params['limit_size_step']))
            f.write('sparsity: {}\n'.format(params['sparsity']))
            f.write('accuracy: {}/{} ({:.2f}%)\n'.format(best, len(test_loader.dataset),
                                                         100. * best / len(test_loader.dataset)))
            L = 0
            sum = 0
            para = 0
            for l in best_switch_count_list:
                program_count = int(l[-1].sum() - args.sparsity * l[-1].numel())
                f.write('layer{} average_program_counts: {:.0f}/({}/{})\n'.format(L, program_count / l[-1].numel(),
                                                                                  program_count, l[-1].numel()))
                sum += program_count
                para += args.sparsity * l[-1].numel()
                L += 1
            f.write('program_sum: {}\n'.format(sum))
            f.write('para: {}\n'.format(int(para)))
            f.write('train epoch accuracy and program count:\n')
            for i in acc_list:
                f.write('{:.4f}, {}\n'.format(i[0] / len(test_loader.dataset), int(i[1] - para)))
            f.close()


if __name__ == '__main__':
    main()