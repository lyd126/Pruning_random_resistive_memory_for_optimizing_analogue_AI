
from __future__ import print_function
import argparse
import math
import numpy as np
import copy

import torch
from torch import _VF
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd

args = None
init_bin = True


def load_from_npz(file_dir):
     # Loads data from a .npz file, iterating through its items and aggregating them into a list, which is then returned as a numpy array.
    a = np.load(file_dir, allow_pickle=True)
    r_list = []
    for j in a:
        for i in a[j]:
            r_list.append(i)
    return np.array(r_list)


def make_dataset(dataset):
    # Converts lists of features and labels into a PyTorch TensorDataset, enabling easy batch loading and processing during model training and evaluation.
    feature_list = []
    label_list = []
    for i in dataset:
        feature_list.append(np.reshape(i['feature'], newshape=[23, 20]))
        label_list.append(np.reshape(i['label'], newshape=1))
    dataset = TensorDataset((torch.from_numpy(np.array(feature_list)).float()),
                            (torch.from_numpy(np.array(label_list)).long()))

    return dataset


def quantization(m, bit, low=0, high=0, mode='un-uni', after_relu=False):
    # Applies quantization to a tensor, either uniformly or non-uniformly, based on the specified mode, bit width, and optionally, post-ReLU condition.
    if mode == 'uni':
        if after_relu:
            low = 0
        quan = torch.tensor([low + i * ((high - low) / (2 ** bit - 1)) for i in range(2 ** bit)]).cuda()
    else:
        m_temp = m.clone()
        m_temp = m_temp.reshape(-1)
        m_temp = torch.sort(m_temp)[0]
        if after_relu:
            p = np.arange(0.5 / (2 ** bit - 1), 1, 1 / (2 ** bit - 1))
            zero_num = torch.where(m_temp == 0)[0].shape[0]
            quan = m_temp[[0] + list((np.round(p * (m_temp.shape[0] - zero_num)) + zero_num).astype(np.int32))]
        else:
            p = np.arange(0.5 / 2 ** bit, 1, 1 / 2 ** bit)
            quan = m_temp[list((np.round(p * m_temp.shape[0])).astype(np.int32))]
    for i in range(1, quan.shape[0] - 1):
        m = torch.where(((m > (quan[i] + quan[i - 1]) / 2) & (m <= (quan[i] + quan[i + 1]) / 2)), quan[i], m)
    m = torch.where(m <= (quan[0] + quan[1]) / 2, quan[0], m)
    m = torch.where(m > (quan[-2] + quan[-1]) / 2, quan[-1], m)

    return m


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class SupermaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Extends the standard nn.Conv2d to include dynamic sparsity via supermasks, with optional binary initialization 
        # of weights and turning off weight gradients to simulate a fixed, sparse structure.
        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        self.init_bin = init_bin
        # NOTE: initialize the weights like this.
        if self.init_bin:
            fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
            # scale:
            fan = fan * (1 - 0.5)
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            self.weight.data = self.weight.data.sign() * std
        else:
            nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.subnet_old = torch.ones_like(self.weight.data).cuda()
        self.score_old = self.scores.data.clone()
        self.weight.requires_grad = False

    def forward(self, x, switch_count=None):
        subnet = GetSubnet.apply(self.scores.abs(), args.sparsity)
        self.subnet_old = subnet.data
        self.score_old = self.scores.data.clone()
        if switch_count is not None:
            switch_count[-1] += torch.where(subnet != switch_count[0], 1.0, 0.0).cuda()
            switch_count[0] = subnet.data.clone()
            w = self.weight * subnet
            x = F.conv2d(
                x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
            return x, switch_count
        else:
            w = self.weight * subnet
            x = F.conv2d(
                x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
            return x


class SupermaskRNN(nn.RNNBase):
    def __init__(self, *args, **kwargs):
        super(SupermaskRNN, self).__init__(*args, **kwargs)
        # Initialize score parameters for input-to-hidden and hidden-to-hidden weights
        # These scores determine which weights are kept active in the supermask
        self.ih_scores = nn.Parameter(torch.Tensor(self.weight_ih_l0.size()))
        nn.init.kaiming_uniform_(self.ih_scores, a=math.sqrt(5))

        self.hh_scores = nn.Parameter(torch.Tensor(self.weight_hh_l0.size()))
        nn.init.kaiming_uniform_(self.hh_scores, a=math.sqrt(5))

        self.init_bin = init_bin
        # NOTE: initialize the weights like this.
        if self.init_bin:
            ih_fan = nn.init._calculate_correct_fan( self.weight_ih_l0, "fan_in")
            # scale:
            ih_fan = ih_fan * (1 - 0.5)
            ih_gain = nn.init.calculate_gain("relu")
            ih_std = ih_gain / math.sqrt(ih_fan)
            self.weight_ih_l0.data = self.weight_ih_l0.data.sign() * ih_std

            hh_fan = nn.init._calculate_correct_fan( self.weight_hh_l0, "fan_in")
            # scale:
            hh_fan = hh_fan * (1 - 0.5)
            hh_gain = nn.init.calculate_gain("relu")
            hh_std = hh_gain / math.sqrt(hh_fan)
            self.weight_hh_l0.data = self.weight_hh_l0.data.sign() * hh_std
        else:
            nn.init.kaiming_normal_(self.weight_ih_l0, mode="fan_in", nonlinearity="relu")
            nn.init.kaiming_normal_(self.weight_hh_l0, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight_ih_l0.requires_grad = False
        self.weight_hh_l0.requires_grad = False

        self.ih_subnet_old = torch.ones_like(self.weight_ih_l0.data).cuda()
        self.ih_score_old = self.ih_scores.data.clone()
        self.hh_subnet_old = torch.ones_like(self.weight_hh_l0.data).cuda()
        self.hh_score_old = self.hh_scores.data.clone()

    def forward(self, input, hx=None, switch_count_ih=None, switch_count_hh=None):  # noqa: F811
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

        weight_ih_l0_subnet = GetSubnet.apply(self.ih_scores.abs(), args.sparsity)
        weight_hh_l0_subnet = GetSubnet.apply(self.hh_scores.abs(), args.sparsity)

        if switch_count_ih is not None:
            self.ih_subnet_old = weight_ih_l0_subnet.data
            self.ih_score_old = self.ih_scores.data.clone()
            self.hh_subnet_old = weight_hh_l0_subnet.data
            self.hh_score_old = self.hh_scores.data.clone()

            switch_count_ih[-1] += torch.where(weight_ih_l0_subnet != switch_count_ih[0], 1.0, 0.0).cuda()
            switch_count_ih[0] = weight_ih_l0_subnet.data.clone()
            switch_count_hh[-1] += torch.where(weight_hh_l0_subnet != switch_count_hh[0], 1.0, 0.0).cuda()
            switch_count_hh[0] = weight_hh_l0_subnet.data.clone()

        w_l = [weight_ih_l0_subnet * self._flat_weights[0], weight_hh_l0_subnet * self._flat_weights[1]]

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


class SupermaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        self.init_bin = init_bin
        # NOTE: initialize the weights like this.
        if self.init_bin:
            fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
            # scale:
            fan = fan * (1 - 0.5)
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            self.weight.data = self.weight.data.sign() * std
        else:
            nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.subnet_old = torch.ones_like(self.weight.data).cuda()
        self.score_old = self.scores.data.clone()
        self.weight.requires_grad = False

    def forward(self, x, switch_count=None):
        subnet = GetSubnet.apply(self.scores.abs(), args.sparsity)
        self.subnet_old = subnet.data
        self.score_old = self.scores.data.clone()
        if switch_count is not None:
            switch_count[-1] += torch.where(subnet != switch_count[0], 1.0, 0.0).cuda()
            switch_count[0] = subnet.data.clone()
            w = self.weight * subnet
            return F.linear(x, w, self.bias), switch_count
        else:
            w = self.weight * subnet
            return F.linear(x, w, self.bias)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SupermaskConv(1, 64, [3, 2], [2, 1], bias=False)
        self.c_bn1 = nn.BatchNorm2d(64, affine=False)
        self.conv2 = SupermaskConv(64, 32, [3, 2], [2, 1], bias=False)
        self.c_bn2 = nn.BatchNorm2d(32, affine=False)
        self.rnn = SupermaskRNN(mode='RNN_RELU', input_size=32, hidden_size=128, num_layers=1, bias=False, batch_first=False)
        # self.dropout = nn.Dropout2d(0.25)
        self.fc1 = SupermaskLinear(128, 256, bias=False)
        self.fc2 = SupermaskLinear(256, 10, bias=False)

    def forward(self, x, qb=0, switch_count_list=None):
        if switch_count_list:
            x = x.unsqueeze(dim=1)
            x.data = quantization(x.data, qb, 0.0, 1.0, 'uni')
            x, switch_count_list[0] = self.conv1(x, switch_count_list[0])
            x = F.max_pool2d(x, 2)
            x = self.c_bn1(x)
            x = F.relu(x)
            x.data = quantization(x.data, qb, after_relu=True)
            x, switch_count_list[1] = self.conv2(x, switch_count_list[1])
            x = F.max_pool2d(x, 2)
            x = self.c_bn2(x)
            x = x.squeeze(dim=2)
            x.data = quantization(x.data, qb)
            x = torch.transpose(x, -2, -1)
            x = torch.transpose(x, 0, 1)
            h_t = torch.zeros(self.rnn.num_layers, x.shape[1], self.rnn.hidden_size, dtype=x.dtype, device=x.device)
            for l in range(x.shape[0]):
                h_t, switch_count_list[2], switch_count_list[3] = self.rnn(x[l].unsqueeze(dim=0), h_t, switch_count_list[2], switch_count_list[3])
                if l:
                    h_all = torch.cat((h_all, h_t), dim=0)
                else:
                    h_all = h_t
                if l != x.shape[0] - 1:
                    h_t = quantization(h_t, qb, after_relu=True)
            aver_h = torch.mean(h_all, dim=0).reshape([x.shape[1], self.rnn.hidden_size])
            x = aver_h
            x.data = quantization(x.data, qb, after_relu=True)
            x, switch_count_list[4] = self.fc1(x, switch_count_list[4])
            x = F.relu(x)
            # x = self.dropout(x)
            x.data = quantization(x.data, qb, after_relu=True)
            x, switch_count_list[5] = self.fc2(x, switch_count_list[5])
            output = F.log_softmax(x, dim=1)
            return output, switch_count_list
        else:
            x = x.unsqueeze(dim=1)
            x.data = quantization(x.data, qb, 0.0, 1.0, 'uni')
            x = self.conv1(x)
            x = F.max_pool2d(x, 2)
            x = self.c_bn1(x)
            x = F.relu(x)
            x.data = quantization(x.data, qb, after_relu=True)
            x = self.conv2(x)
            x = F.max_pool2d(x, 2)
            x = self.c_bn2(x)
            x = x.squeeze(dim=2)
            x.data = quantization(x.data, qb)
            x = torch.transpose(x, -2, -1)
            x = torch.transpose(x, 0, 1)
            h_t = torch.zeros(self.rnn.num_layers, x.shape[1], self.rnn.hidden_size, dtype=x.dtype, device=x.device)
            for l in range(x.shape[0]):
                h_t = self.rnn(x[l].unsqueeze(dim=0), h_t)
                if l:
                    h_all = torch.cat((h_all, h_t), dim=0)
                else:
                    h_all = h_t
                if l != x.shape[0] - 1:
                    h_t = quantization(h_t, qb, after_relu=True)
            aver_h = torch.mean(h_all, dim=0).reshape([x.shape[1], self.rnn.hidden_size])
            x = aver_h
            x.data = quantization(x.data, qb, after_relu=True)
            x = self.fc1(x)
            x = F.relu(x)
            # x = self.dropout(x)
            x.data = quantization(x.data, qb, after_relu=True)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output


def train(model, qb, switch_count_list, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.squeeze(dim=1).to(device)
        optimizer.zero_grad()
        output, switch_count_list = model(data, qb=qb, switch_count_list=switch_count_list)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    return switch_count_list


def Test(model, qb, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.squeeze(dim=1).to(device)
            output = model(data, qb=qb)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    return correct


def main():
    global args

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=30, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
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
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--data', type=str, default='data', help='Location to store data')
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='how sparse is each layer')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_dataset = load_from_npz("train_features.npz")
    test_dataset = load_from_npz("test_features.npz")

    train_loader = DataLoader(make_dataset(train_dataset),
                               batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)
    test_loader = DataLoader(make_dataset(test_dataset),
                               batch_size=args.test_batch_size, drop_last=True, shuffle=False, **kwargs)

    model = Net().to(device)
    # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important!
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss().to(device)
    best = 0
    switch_count_list = []
    for sl in [p for p in model.parameters() if p.requires_grad]:
        switch_count_list.append([
            torch.ones_like(sl.data).cuda(),
            torch.zeros_like(sl.data).cuda()
        ])

    acc_list = []

    for epoch in range(1, args.epochs + 1):
        switch_count_list = train(model, args.q_bit, switch_count_list, device, train_loader, optimizer, criterion, epoch)
        correct = Test(model, args.q_bit, device, test_loader)
        acc_list.append(correct)
        if best < correct:
            best = correct
            best_dict = copy.deepcopy(model.state_dict())
            print("best accuracy!\n")

        scheduler.step()

    print("best accuracy: {:.2f}%\n".format(100. * best / len(test_loader.dataset)))

    if args.save_model:
        torch.save(best_dict, "spoken_digit_software_ep_{}.pt".format(best))
        with open("spoken_digit_software_ep_{}.txt".format(best), 'w') as f:
            f.write('arch:\n 64-32-128-256-10\n')
            f.write('training_epochs: {}\n'.format(args.epochs))
            f.write('batch_size: {}\n'.format(args.batch_size))
            f.write('q_bit: {}\n'.format(args.q_bit))
            f.write('accuracy: {}/{} ({:.0f}%)\n'.format(best, len(test_loader.dataset), 100. * best / len(test_loader.dataset)))
            L = 0
            for l in switch_count_list:
                program_count = int(l[-1].sum() - args.sparsity * l[-1].numel())
                f.write('layer{} average_program_counts: {:.0f}/({}/{})\n'.format(L, program_count / l[-1].numel(),
                                                                                  program_count, l[-1].numel()))
                L += 1
            f.write('train epoch accuracy:\n')
            for i in acc_list:
                f.write('{:.4f}\n'.format(i / len(test_loader.dataset)))
            f.close()


if __name__ == '__main__':
    main()
