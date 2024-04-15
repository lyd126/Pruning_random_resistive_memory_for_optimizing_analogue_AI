# Import necessary libraries
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

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

# Initialize variables and parameters for later use
args = None
init_bin = True

# Define a dictionary for model layer sizes
model_state_dict = {
    'conv1_weight': [64, 1, 3, 3],
    'conv2_weight': [16, 64, 3, 3],
    'fc1_weight': [128, 400],
    'fc2_weight': [10, 128],
}


def quantization(m, bit, after_relu=False):
    # Quantize weights function, for quantization in software
    out = m.clone()
    scale = (out.max() - out.min()) / (2 ** bit - 1)
    if after_relu:
        out = torch.quantize_per_tensor(out, scale=scale, zero_point=0, dtype=torch.quint8)
        out = (out.int_repr() * scale).float().cuda()
    else:
        zp_max = out.max() // scale
        zp_min = out.min() // scale
        zp = zp_min + 2 ** (bit - 1) - zp_max - 1
        out = torch.quantize_per_tensor((out - out.min()), scale=scale, zero_point=zp, dtype=torch.qint32)
        out = ((out.int_repr() - zp + zp_min) * scale).float().cuda()

    return out


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
    """
    Custom convolutional layer that applies 'supermasks' to its weights.
    Supermasks are binary masks that are applied to the weights of the convolutional layer
    to selectively activate certain weights during the forward pass. This concept is often used in
    network pruning and sparse neural networks.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the SupermaskConv layer.
        """
        super().__init__(*args, **kwargs)  # Initialize the parent Conv2d class

        # Initialize the scores for each weight. These scores determine the supermask values during training.
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))  # Use He initialization for scores

        self.init_bin = kwargs.get('init_bin', False)  # Binary initialization flag
        if self.init_bin:
            # If binary initialization is enabled, initialize weights based on fan-in and adjust for sparsity
            fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
            fan *= (1 - 0.5)  # Adjust fan for assumed sparsity
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            self.weight.data = self.weight.data.sign() * std  # Apply binary initialization
        else:
            # Standard weight initialization using Kaiming normal method
            nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # Prepare variables for tracking the old state of the network
        self.subnet_old = torch.ones_like(self.weight.data).cuda()
        self.score_old = self.scores.data.clone()
        self.weight.requires_grad = False  # Turn off gradients for the weights

    def forward(self, x, switch_count=None):
        """
        Defines the forward pass of the layer.

        Args:
            x: Input tensor.
            switch_count: Optional tensor to count the number of weight switches (for analysis purposes).

        Returns:
            The convolved output tensor, and optionally, the switch count.
        """
        # Calculate the subnet based on absolute scores, enforcing the desired sparsity level
        subnet = GetSubnet.apply(self.scores.abs(), args.sparsity)
        self.subnet_old = subnet.data  # Save the current subnet for later reference

        if switch_count is not None:
            # If switch_count is provided, update it based on the changes in the subnet
            switch_count[-1] += torch.where(subnet != switch_count[0], 1.0, 0.0).cuda()
            switch_count[0] = subnet.data.clone()

        # Apply the subnet mask to the weights
        w = self.weight * subnet
        # Perform the convolution operation with the masked weights
        x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # Update the score_old variable for next iteration
        self.score_old = self.scores.data.clone()

        if switch_count is not None:
            return x, switch_count
        else:
            return x


class SupermaskLinear(nn.Linear):
    """
    Custom linear (fully connected) layer that applies 'supermasks' to its weights.
    A 'supermask' is a binary mask that is applied to the weights of the linear layer to selectively
    activate certain weights during the forward pass, which can be useful for creating sparse neural networks.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the SupermaskLinear layer.
        """
        super().__init__(*args, **kwargs)  # Initialize the parent Linear class

        # Initialize the scores for each weight. These scores determine the supermask values.
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))  # Use He initialization for scores

        self.init_bin = kwargs.get('init_bin', False)  # Binary initialization flag
        if self.init_bin:
            # Binary initialization: adjust weights based on fan-in, considering assumed sparsity
            fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
            fan *= (1 - 0.5)  # Adjust for assumed sparsity
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            self.weight.data = self.weight.data.sign() * std  # Apply binary initialization
        else:
            # Standard weight initialization using Kaiming normal method
            nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # Turn off gradients for the weights as they will be masked
        self.subnet_old = torch.ones_like(self.weight.data).cuda()  # Placeholder for subnet tracking
        self.score_old = self.scores.data.clone()  # Keep a copy of scores for tracking changes
        self.weight.requires_grad = False  # Disable weight updates during optimization

    def forward(self, x, switch_count=None):
        """
        Defines the forward pass with an optional switch count for tracking changes in the supermask.

        Args:
            x: Input tensor.
            switch_count: Optional tensor to count the number of weight switches, useful for analysis.

        Returns:
            The linearly transformed output tensor, and optionally, the switch count.
        """
        # Calculate the subnet (supermask) based on absolute scores, enforcing sparsity
        subnet = GetSubnet.apply(self.scores.abs(), args.sparsity)
        self.subnet_old = subnet.data  # Save the current subnet for reference

        if switch_count is not None:
            # Update switch count based on changes in the subnet
            switch_count[-1] += torch.where(subnet != switch_count[0], 1.0, 0.0).cuda()
            switch_count[0] = subnet.data.clone()

        # Apply the subnet mask to the weights
        w = self.weight * subnet
        self.score_old = self.scores.data.clone()  # Update scores for next iteration

        if switch_count is not None:
            return F.linear(x, w, self.bias), switch_count  # Apply masked weights in linear operation
        else:
            return F.linear(x, w, self.bias)  # Linear operation with masked weights


# Note: Although not used here, non-affine normalization means normalization without learned parameters
# (e.g., mean subtraction and division by standard deviation, without scaling and shifting).

class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)


class Net(nn.Module):
    """
    A neural network class that integrates supermask convolutional and linear layers for sparse neural networks,
    alongside standard max pooling, batch normalization, ReLU activations, and quantization for efficient processing.
    """

    def __init__(self):
        """
        Initializes the network with two supermask convolutional layers followed by two supermask linear layers.
        Uses parameters from a provided state dictionary (`model_state_dict`) for layer dimensions.
        Batch normalization layers are configured without affine parameters.
        """
        super(Net, self).__init__()

        # Max pooling layer to reduce spatial dimensions of the input
        self.maxpooling = nn.MaxPool2d(2)

        # First convolutional layer with supermask
        self.conv1 = SupermaskConv(model_state_dict['conv1_weight'][1],
                                   model_state_dict['conv1_weight'][0],
                                   model_state_dict['conv1_weight'][2], 1, bias=False)

        # Batch normalization for the output of the first convolutional layer
        self.c_bn1 = nn.BatchNorm2d(model_state_dict['conv1_weight'][0], affine=False)

        # Second convolutional layer with supermask
        self.conv2 = SupermaskConv(model_state_dict['conv2_weight'][1],
                                   model_state_dict['conv2_weight'][0],
                                   model_state_dict['conv2_weight'][2], 1, bias=False)

        # Batch normalization for the output of the second convolutional layer
        self.c_bn2 = nn.BatchNorm2d(model_state_dict['conv2_weight'][0], affine=False)

        # First fully connected (linear) layer with supermask
        self.fc1 = SupermaskLinear(model_state_dict['fc1_weight'][-1],
                                   model_state_dict['fc1_weight'][0], bias=False)

        # Second fully connected (linear) layer with supermask
        self.fc2 = SupermaskLinear(model_state_dict['fc2_weight'][-1],
                                   model_state_dict['fc2_weight'][0], bias=False)

    def forward(self, x, qb=0, switch_count_list=None):
        """
        Defines the forward pass of the network with optional quantization and tracking of weight switches.

        Args:
            x: Input tensor.
            qb: Quantization bits. If greater than 0, applies quantization.
            switch_count_list: Optional list to track the number of switches in supermask layers.

        Returns:
            The network's output, and optionally, the updated switch count list.
        """
        x = self.maxpooling(x)  # Apply max pooling
        x.data = quantization(x.data, qb)  # Optionally quantize the data

        # Process through the first convolutional layer
        if switch_count_list:
            x, switch_count_list[0] = self.conv1(x, switch_count_list[0])
        else:
            x = self.conv1(x)

        x = self.c_bn1(x)  # Apply batch normalization
        x = F.relu(x)  # Apply ReLU activation
        x.data = quantization(x.data, qb, after_relu=True)  # Optionally quantize the data after ReLU

        # Process through the second convolutional layer
        if switch_count_list:
            x, switch_count_list[1] = self.conv2(x, switch_count_list[1])
        else:
            x = self.conv2(x)

        x = F.max_pool2d(x, 2)  # Apply max pooling
        x = self.c_bn2(x)  # Apply batch normalization
        x = F.relu(x)  # Apply ReLU activation
        x.data = quantization(x.data, qb, after_relu=True)  # Optionally quantize the data after ReLU
        x = torch.flatten(x, 1)  # Flatten the tensor for the fully connected layers

        # Process through the first fully connected layer
        if switch_count_list:
            x, switch_count_list[2] = self.fc1(x, switch_count_list[2])
        else:
            x = self.fc1(x)

        x = F.relu(x)  # Apply ReLU activation
        x.data = quantization(x.data, qb, after_relu=True)  # Optionally quantize the data after ReLU

        # Process through the second fully connected layer
        if switch_count_list:
            x, switch_count_list[3] = self.fc2(x, switch_count_list[3])
        else:
            x = self.fc2(x)

        output = F.log_softmax(x, dim=1)  # Apply log softmax to get the final output
        if switch_count_list:
            return output, switch_count_list
        else:
            return output


def train(model, switch_count_list, qb, device, train_loader, optimizer, criterion, epoch):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model to train.
        switch_count_list: A list to track the number of switches in supermask layers.
        qb: The number of quantization bits to use in quantization.
        device: The device (CPU or GPU) to perform the training on.
        train_loader: DataLoader for the training data.
        optimizer: The optimization algorithm.
        criterion: The loss function.
        epoch: The current epoch number.

    Returns:
        Updated switch_count_list after the training epoch.
    """
    model.train()  # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):  # Loop through the data batches
        data, target = data.to(device), target.to(device)  # Move data to the specified device

        optimizer.zero_grad()  # Clear gradients for the next train step

        output, switch_count_list = model(data, qb=qb, switch_count_list=switch_count_list)  # Forward pass
        loss = criterion(output, target)  # Compute the loss
        loss.backward()  # Compute gradients of the loss wrt the parameters
        optimizer.step()  # Update the parameters

        if batch_idx % args.log_interval == 0:  # Logging
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    return switch_count_list


def Test(model, qb, device, test_loader):
    """
    Tests the model on the test dataset.

    Args:
        model: The neural network model to test.
        qb: The number of quantization bits to use in quantization.
        device: The device (CPU or GPU) to perform the testing on.
        test_loader: DataLoader for the test data.

    Returns:
        The number of correct predictions.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0  # Counter for correct predictions
    with torch.no_grad():  # Disable gradient computation
        for data, target in test_loader:  # Loop through the data batches
            data, target = data.to(device), target.to(device)  # Move data to the specified device

            output = model(data, qb=qb)  # Forward pass
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct


def main():
    global args

    params = {
        'epochs': 65,
        'q_bit': 4,
        'lr': 0.1,
        'momentum': 0.9,
        'wd': 0.0005,
        'batch_size': 60,
        'sparsity': 0.5,
    }

    # Training settings
    parser = argparse.ArgumentParser(description="Pytorch CNN with Software TO")

    parser.add_argument('--batch-size', type=int, default=60, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--q_bit', type=int, default=4, metavar='N', help='bit of quantized input (default: 3)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='Momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=0.0005, metavar='M', help='Weight decay (default: 0.0005)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    parser.add_argument('--data', type=str, default='../data', help='Location to store data')
    parser.add_argument('--sparsity', type=float, default=0.5, help='how sparse is each layer')

    args = parser.parse_args()
    # Override sparsity with predefined value
    args.sparsity = params['sparsity']
    # Determine if CUDA is available and set the device accordingly
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    # DataLoader arguments like num_workers and pin_memory are set based on whether CUDA is used
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Load and preprocess the FashionMNIST dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(os.path.join(args.data, 'fashionmnist'), train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(os.path.join(args.data, 'fashionmnist'), train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)
    # Instantiate the model and move it to the appropriate device
    model = Net().to(device)
    # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important!
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=params['lr'],
        momentum=params['momentum'],
        weight_decay=params['wd'],
    )
    # Loss function and scheduler for adjusting the learning rate
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=params['epochs'])
    best = 0
    switch_count_list = []
    # Initialize switch_count_list for each trainable parameter
    for sl in [p for p in model.parameters() if p.requires_grad]:
        switch_count_list.append([
            torch.ones_like(sl.data).cuda(),
            torch.zeros_like(sl.data).cuda()
        ])

    acc_list = []
    # Main training loop
    for epoch in range(1, params['epochs'] + 1):
        switch_count_list = train(model, switch_count_list, params['q_bit'], device, train_loader, optimizer, criterion,
                                  epoch)
        correct = Test(model, params['q_bit'], device, test_loader)
        acc_list.append(correct)
        if best <= correct:
            best = correct
            best_dict = copy.deepcopy(model.state_dict())
            print("best accuracy!\n")

        scheduler.step()
    print("best accuracy: {:.2f}%\n".format(100. * best / len(test_loader.dataset)))

    # Obtain the Current Time
    now = datetime.datetime.now()
    # Convert the Timestamp to a String
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")

    if args.save_model:
        torch.save(best_dict, "fashionmnist_software_ep_program_{}_".format(best) + timestamp_str + ".pt")
        shape_list = []
        for i in best_dict.keys():
            if "weight" in i:
                shape_list.append([k for k in best_dict[i].shape])
        with open("fashionmnist_software_ep_program_{}_".format(best) + timestamp_str + ".txt", 'w') as f:
            f.write('arch:\n')
            for s in shape_list:
                arch = str(s[0])
                for t in s[1:]:
                    arch = arch + '_{}'.format(t)
                f.write('\t' + arch + '\n')
            f.write('training_epochs: {}\n'.format(params['epochs']))
            f.write('batch_size: {}\n'.format(params['batch_size']))
            f.write('input_bit: {}\n'.format(params['q_bit']))
            f.write('sparsity: {}\n'.format(params['sparsity']))
            f.write('accuracy: {}/{} ({:.2f}%)\n'.format(best, len(test_loader.dataset),
                                                         100. * best / len(test_loader.dataset)))
            L = 0
            sum = 0
            for l in switch_count_list:
                program_count = int(l[-1].sum() - args.sparsity * l[-1].numel())
                f.write('layer{} average_program_counts: {:.0f}/({}/{})\n'.format(L, program_count / l[-1].numel(),
                                                                                  program_count, l[-1].numel()))
                sum += program_count
                L += 1
            f.write('program_sum: {}\n'.format(sum))
            f.write('train epoch accuracy:\n')
            for i in acc_list:
                f.write('{:.4f}\n'.format(i / len(test_loader.dataset)))
            f.close()


if __name__ == '__main__':
    main()