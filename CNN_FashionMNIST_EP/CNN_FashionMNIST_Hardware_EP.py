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

# Set CUDA device to GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Initialize variables and parameters for later use
args = None
limit_size_start = None
limit_size_stop = None
limit_size_step = None
limit_size = None
p_noise = 0.15

def inject_noise(w):
    # Add noise to weights function, for simulating hardware imperfections
    w = w + 1e-9
    sigma = p_noise / 100 * abs(w)
    w_n = np.random.normal(loc=w, scale=sigma)
    p2s = w + 2 * sigma
    n2s = w - 2 * sigma
    w_n = np.where(w_n > p2s, p2s, w_n)
    w_n = np.where(w_n < n2s, n2s, w_n)
    w_n = np.where(w_n == 0.0, 1e-9, w_n)
    return w_n

def quantization(m, bit, after_relu=False):
    # Quantize weights function, for simulating quantization in hardware
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

# Since not everyone has the RRAM chip, we have provided pre-read RRAM data for your convenience.
hardware_weight = torch.load('RRAM_hardware_weight_1000.pt', map_location='cpu')
hardware_weight_n = copy.deepcopy(hardware_weight)
for i in hardware_weight_n:
    hardware_weight_n[i] = inject_noise(hardware_weight_n[i])

weight_list_one = []
weight_list_zero = []
weight_list_one_n = []
weight_list_zero_n = []

# Define a dictionary for model layer sizes
model_state_dict = {
    'conv1_weight': [64, 1, 3, 3],
    'conv2_weight': [16, 64, 3, 3],
    'fc1_weight':   [128, 400],
    'fc2_weight':   [10, 128],
}

# Prepare hardware weights based on model layer sizes
# Logic to handle hardware weight transformations based on "set" and "reset" operations
# Further processing of weights to align with layer sizes
index = 0
half_size = hardware_weight["set"].shape[-1] // 2
for i in model_state_dict.keys():
    if "weight" in i:
        size = 1
        for j in model_state_dict[i]:
            size *= j
        weight_list_one.append((hardware_weight["set"][:, index: index + size] - hardware_weight["set"][:, half_size + index: half_size + index + size]).reshape(
            [hardware_weight["set"].shape[0]] + model_state_dict[i]))
        weight_list_zero.append((hardware_weight["reset"][:, index: index + size] - hardware_weight["set"][:, half_size + index: half_size + index + size]).reshape(
            [hardware_weight["set"].shape[0]] + model_state_dict[i]))
        weight_list_one_n.append((hardware_weight_n["set"][:, index: index + size] - hardware_weight_n["set"][:, half_size + index: half_size + index + size]).reshape(
            [hardware_weight_n["set"].shape[0]] + model_state_dict[i]))
        weight_list_zero_n.append((hardware_weight_n["reset"][:, index: index + size] - hardware_weight_n["set"][:, half_size + index: half_size + index + size]).reshape(
            [hardware_weight_n["set"].shape[0]] + model_state_dict[i]))
        index += size
hardware_weight = []
hardware_weight_n = []

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

        #limit
        if mask_old.sum() != mask_old.numel():
            d_s = (score_old - scores.clone()).abs() / score_old
            out = torch.where(d_s < limit_size, mask_old, out)
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None, None, None


class SupermaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        # Call the constructor of the parent class (nn.Conv2d) to initialize the convolutional layer.
        super().__init__(*args, **kwargs)

        # Initialize scores for each weight in the convolutional layer. These scores determine
        # the importance of each weight and whether it should be included in the subnet.
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))  # Use Kaiming uniform initialization for the scores.

        # Load hardware-specific weights. These weights are adjusted for hardware imperfections
        # and are used to simulate hardware behavior in the neural network.
        for i in range(len(weight_list_one)):
            if weight_list_one[i][0].shape == self.weight.data.shape:
                self.hardware_weight_one = torch.from_numpy(weight_list_one[i].copy()).abs().float().cuda()
                self.hardware_weight_zero = torch.from_numpy(weight_list_zero[i].copy()).abs().float().cuda()
                self.hardware_weight_one_n = torch.from_numpy(weight_list_one_n[i].copy()).abs().float().cuda()
                self.hardware_weight_zero_n = torch.from_numpy(weight_list_zero_n[i].copy()).abs().float().cuda()
                break

        # Initialize standard deviation (std) and scale for weight normalization, based on fan-in.
        fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
        fan = fan * (1 - 0.5)  # Adjust the fan value for the scaling calculation.
        gain = nn.init.calculate_gain("relu")  # Calculate the gain for ReLU activation.
        self.std = gain / math.sqrt(fan)
        self.scale = self.hardware_weight_one.mean() / self.std

        # Initialize additional parameters for tracking and manipulating weights.
        self.sign = self.weight.data.sign().cuda()  # Store the sign of each weight.
        self.subnet_old = torch.ones_like(self.weight.data).cuda()  # Keep the previous subnet mask.
        self.score_old = self.scores.data.clone()  # Keep the previous scores.
        self.weight.requires_grad = False  # Disable gradient computation for weights, as they are managed manually.

    def forward(self, x, count, switch_count=None):
        # Compute the subnet mask based on the current scores and sparsity parameter.
        subnet = GetSubnet.apply(self.scores.abs(), args.sparsity, self.subnet_old, self.score_old.abs())

        # If switch_count is provided, it's used to track changes in the subnet over time.
        if switch_count is not None:
            switch_count[-1] += torch.where(subnet != switch_count[0], 1.0, 0.0).cuda()
            self.subnet_old = subnet.data
            self.score_old = self.scores.data.clone()

            # Calculate the effective weights based on the subnet mask and hardware weights.
            # This step also accounts for possible switches between the hardware states.
            w = self.sign * torch.where(subnet != switch_count[0],
                                        self.hardware_weight_one_n[count] * subnet + self.hardware_weight_zero_n[count] * (1 - subnet),
                                        self.hardware_weight_one[count] * subnet + self.hardware_weight_zero[count] * (1 - subnet))

            # Perform the convolution operation with the effective weights.
            x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            switch_count[0] = subnet.data.clone()
            return x / self.scale, switch_count  # Normalize the output and return along with the updated switch_count.
        else:
            # If no switch_count is provided, simply compute the effective weights based on the current subnet mask.
            w = self.sign * (self.hardware_weight_one[count] * subnet + self.hardware_weight_zero[count] * (1 - subnet))
            x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return x / self.scale  # Normalize and return the output.



class SupermaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        # Initialize the linear layer with the provided arguments and keyword arguments.
        super().__init__(*args, **kwargs)

        # Initialize scores for each weight in the linear layer. These scores are used to determine
        # the importance of each weight and whether it should be activated in the subnet.
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        # Use Kaiming uniform initialization for initializing the scores to ensure they start in a reasonable range.
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # Load hardware-specific weights for simulating the behavior of hardware in neural network training.
        # This includes weights for both the "one" state and the "zero" state, as well as their noisy counterparts.
        for i in range(len(weight_list_one)):
            if weight_list_one[i][0].shape == self.weight.data.shape:
                # Copy and convert the hardware weights to tensors that can be used in CUDA operations.
                self.hardware_weight_one = torch.from_numpy(weight_list_one[i].copy()).abs().float().cuda()
                self.hardware_weight_zero = torch.from_numpy(weight_list_zero[i].copy()).abs().float().cuda()
                self.hardware_weight_one_n = torch.from_numpy(weight_list_one_n[i].copy()).abs().float().cuda()
                self.hardware_weight_zero_n = torch.from_numpy(weight_list_zero_n[i].copy()).abs().float().cuda()
                break

        # Initialize standard deviation and scaling factor for weight normalization.
        # This is based on the fan-in mode and adjusted for half the units not being active due to sparsity.
        fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
        fan = fan * (1 - 0.5)  # Adjust the fan value for the expected sparsity.
        gain = nn.init.calculate_gain("relu")  # Calculate gain for ReLU, considering it's the likely activation.
        self.std = gain / math.sqrt(fan)
        self.scale = self.hardware_weight_one.mean() / self.std

        # Additional initializations for managing the subnet and its evolution.
        self.sign = self.weight.data.sign().cuda()  # Store the sign of each weight for later adjustments.
        self.subnet_old = torch.ones_like(self.weight.data).cuda()  # Keep the previous subnet mask.
        self.score_old = self.scores.data.clone()  # Keep the previous scores for tracking changes.
        self.weight.requires_grad = False  # Disable gradient computation for the weights, managed manually.

    def forward(self, x, count, switch_count=None):
        # Apply the subnet mask to the weights based on their scores, creating a dynamic sparsity pattern.
        subnet = GetSubnet.apply(self.scores.abs(), args.sparsity, self.subnet_old, self.score_old.abs())
        
        # If switch_count is provided, track changes in the subnet mask over time for additional control.
        if switch_count is not None:
            # Update the old subnet and score trackers.
            self.subnet_old = subnet.data
            self.score_old = self.scores.data.clone()
            # Count switches in the subnet mask, indicating changes in active weights.
            switch_count[-1] += torch.where(subnet != switch_count[0], 1.0, 0.0).cuda()
            # Calculate effective weights based on the current subnet mask and the hardware-specific weights.
            w = self.sign * torch.where(subnet != switch_count[0],
                                        self.hardware_weight_one_n[count] * subnet + self.hardware_weight_zero_n[count] * (1 - subnet),
                                        self.hardware_weight_one[count] * subnet + self.hardware_weight_zero[count] * (1 - subnet))
            switch_count[0] = subnet.data.clone()
            # Normalize the output by the scale factor and return it along with the updated switch_count.
            return F.linear(x, w, self.bias) / self.scale, switch_count
        else:
            # If no switch_count is provided, directly compute the effective weights for the current mask.
            w = self.sign * (self.hardware_weight_one[count] * subnet + self.hardware_weight_zero[count] * (1 - subnet))
            # Perform the linear operation using the effective weights and normalize the output.
            return F.linear(x, w, self.bias) / self.scale



class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)


class Net(nn.Module):
    def __init__(self):
        # Initialize the parent class (nn.Module) constructor.
        super(Net, self).__init__()
        
        # Define max pooling layer to reduce the spatial dimensions of the input tensor.
        self.maxpooling = nn.MaxPool2d(2)
        
        # Define the first convolutional layer with dynamic sparsity through supermask.
        # The layer configuration is retrieved from a predefined dictionary (model_state_dict).
        self.conv1 = SupermaskConv(model_state_dict['conv1_weight'][1], model_state_dict['conv1_weight'][0], model_state_dict['conv1_weight'][2], 1, bias=False)
        
        # Define a batch normalization layer for the output of conv1, with no learnable affine parameters.
        self.c_bn1 = nn.BatchNorm2d(model_state_dict['conv1_weight'][0], affine=False)
        
        # Define the second convolutional layer similar to the first, using supermask for dynamic sparsity.
        self.conv2 = SupermaskConv(model_state_dict['conv2_weight'][1], model_state_dict['conv2_weight'][0], model_state_dict['conv2_weight'][2], 1, bias=False)
        
        # Batch normalization for the output of conv2, again with no learnable affine parameters.
        self.c_bn2 = nn.BatchNorm2d(model_state_dict['conv2_weight'][0], affine=False)
        
        # Define two fully connected (linear) layers with dynamic sparsity through supermask.
        self.fc1 = SupermaskLinear(model_state_dict['fc1_weight'][-1], model_state_dict['fc1_weight'][0], bias=False)
        self.fc2 = SupermaskLinear(model_state_dict['fc2_weight'][-1], model_state_dict['fc2_weight'][0], bias=False)

    def forward(self, x, qb=0, count=0, switch_count_list=None):
        # Apply max pooling to the input tensor.
        x = self.maxpooling(x)
        # Quantize the data after pooling according to a specified bit depth (qb).
        x.data = quantization(x.data, qb)
        
        # If a switch_count_list is provided, use it to manage dynamic subnet changes across layers.
        if switch_count_list:
            count = 0  # Reset count if using switch_count_list for tracking dynamic subnet evolution.
            # Process through the first convolutional layer and batch normalization, applying ReLU and quantization.
            x, switch_count_list[0] = self.conv1(x, count, switch_count_list[0])
            x = self.c_bn1(x)
            x = F.relu(x)
            x.data = quantization(x.data, qb, after_relu=True)
            
            # Process through the second convolutional layer, batch normalization, applying ReLU and quantization.
            x, switch_count_list[1] = self.conv2(x, count, switch_count_list[1])
            x = F.max_pool2d(x, 2)
            x = self.c_bn2(x)
            x = F.relu(x)
            x.data = quantization(x.data, qb, after_relu=True)
            
            # Flatten the tensor to prepare for linear layers.
            x = torch.flatten(x, 1)
            
            # Process through the fully connected layers with ReLU and quantization.
            x, switch_count_list[2] = self.fc1(x, count, switch_count_list[2])
            x = F.relu(x)
            x.data = quantization(x.data, qb, after_relu=True)
            x, switch_count_list[3] = self.fc2(x, count, switch_count_list[3])
            output = F.log_softmax(x, dim=1)  # Apply log softmax for the output layer.
            count += 1  # Increment count after processing through the network.
            return output, count, switch_count_list
        else:
            # If no switch_count_list is provided, process through the network without tracking dynamic subnet evolution.
            # The flow remains the same as described above but without updating or using switch_count_list.
            x = self.conv1(x, count)
            x = self.c_bn1(x)
            x = F.relu(x)
            x.data = quantization(x.data, qb, after_relu=True)
            x = self.conv2(x, count)
            x = F.max_pool2d(x, 2)
            x = self.c_bn2(x)
            x = F.relu(x)
            x.data = quantization(x.data, qb, after_relu=True)
            x = torch.flatten(x, 1)
            x = self.fc1(x, count)
            x = F.relu(x)
            x.data = quantization(x.data, qb, after_relu=True)
            x = self.fc2(x, count)
            output = F.log_softmax(x, dim=1)
            count += 1
            return output, count



def train(model, switch_count_list, qb, device, train_loader, optimizer, criterion, epoch):
    # Set the model to training mode. This is important for layers like dropout and batch normalization.
    model.train()

    # Initialize the count for tracking the number of times the network's subnets have been updated.
    count = 0

    # Iterate over the training data loader. Each iteration returns a batch of data and corresponding targets.
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move the data and target tensors to the specified device (CPU or GPU).
        data, target = data.to(device), target.to(device)

        # Reset the optimizer's gradient buffers to zero. Necessary to prevent accumulation from previous iterations.
        optimizer.zero_grad()

        # Pass the input data through the model. This step also updates the switch_count_list if applicable,
        # allowing for dynamic adjustment of the network's active subnets based on predefined criteria.
        output, count, switch_count_list = model(data, qb=qb, count=count, switch_count_list=switch_count_list)

        # Calculate the loss between the model's predictions and the actual targets.
        loss = criterion(output, target)

        # Perform backpropagation: compute the gradient of the loss with respect to model parameters.
        loss.backward()

        # Update the model's parameters based on the gradients calculated by backpropagation.
        optimizer.step()

        # Log training progress information periodically, as defined by `args.log_interval`.
        # This includes the current epoch, batch index, loss value, and limit sizes (a parameter affecting subnet switching).
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLimit sizes: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), limit_size))
    
    # Return the updated switch_count_list for potential use in subsequent training iterations or analysis.
    return switch_count_list



def Test(model, qb, device, test_loader):
    # Switch the model to evaluation mode. This is crucial for layers like dropout and batch normalization,
    # ensuring they behave differently during inference compared to training.
    model.eval()

    # Initialize counters for correct predictions and subnet switches.
    correct = 0
    count = 0

    # Disable gradient computation to save memory and computations, as gradients are not needed for inference.
    with torch.no_grad():
        # Iterate over the test dataset loader.
        for data, target in test_loader:
            # Transfer data and targets to the appropriate device (CPU or GPU).
            data, target = data.to(device), target.to(device)

            # Forward pass: compute the model's predictions for the current batch of test data.
            output, count = model(data, qb=qb, count=count)

            # Determine the predicted class by finding the index of the maximum log-probability.
            pred = output.argmax(dim=1, keepdim=True)

            # Increment the correct predictions counter by the number of correctly predicted instances in the batch.
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Print out the test accuracy and the number of hardware read operations simulated during the test phase.
    print('\nHardware read times: {}, Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        count, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # Return the total number of correct predictions made over the test dataset.
    return correct



def main():
    
    # Global variables for accessing outside the function scope if needed.
    global args
    global limit_size_start
    global limit_size_stop
    global limit_size_step
    global limit_size
    global p_noise
    
    # Parameters for the training and model configuration, including epochs, learning rate, batch size, etc.
    params = {
        'epochs': 65,
        'q_bit': 4,
        'lr': 0.1,
        'momentum': 0.9,
        'wd': 0.0005,
        'batch_size': 60,
        'p_noise': p_noise,
        'limit_size_start' : 0.01,
        'limit_size_stop' : 0.007,
        'limit_size_step' : 15,
        'sparsity' : 0.5,
    }

    # Parsing command line arguments for customizable training settings.
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=180, metavar='N',help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',help='number of epochs to train (default: 14)')
    parser.add_argument('--q_bit', type=int, default=4, metavar='N',help='bit of quantized input (default: 3)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',help='Momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=0.0005, metavar='M',help='Weight decay (default: 0.0005)')
    parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,help='For Saving the current Model')
    parser.add_argument('--data', type=str, default='../data', help='Location to store data')
    parser.add_argument('--sparsity', type=float, default=0.49,help='how sparse is each layer')
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.sparsity = params['sparsity']
    limit_size_start = params['limit_size_start']
    limit_size_stop = params['limit_size_stop']
    limit_size_step = params['limit_size_step']
    limit_size = limit_size_start
    
    # Determining whether CUDA is available and setting the device accordingly.
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    # DataLoader configuration for both training and test datasets.
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(os.path.join(args.data, 'fashionmnist'), train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=params['batch_size'], shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(os.path.join(args.data, 'fashionmnist'), train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # Initializing the model and moving it to the appropriate compute device.
    model = Net().to(device)
    
    # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important!
    # Setting up the optimizer, loss criterion, and learning rate scheduler.
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=params['lr'],
        momentum=params['momentum'],
        weight_decay=params['wd'],
    )
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=params['epochs'])
    
    # Variables to track the best model and switch counts for analyzing dynamic subnet behavior.
    best = 0
    switch_count_list = []
    for sl in [p for p in model.parameters() if p.requires_grad]:
        switch_count_list.append([
            torch.ones_like(sl.data).cuda(),
            torch.zeros_like(sl.data).cuda()
        ])

    acc_list = []

    # Training and testing loop across specified number of epochs.
    for epoch in range(1, params['epochs'] + 1):
         # Training phase with dynamic subnet and quantization.
        switch_count_list = train(model, switch_count_list, params['q_bit'], device, train_loader, optimizer, criterion, epoch)
        # Testing phase to evaluate model accuracy.
        correct = Test(model, params['q_bit'], device, test_loader)
        acc_list.append(correct)
        # Saving the best model based on accuracy.
        if best <= correct:
            best = correct
            best_dict = copy.deepcopy(model.state_dict())
            # Adjusting the limit size for dynamic subnet adjustment if needed.
            limit_size = limit_size - (limit_size_start - limit_size_stop) / limit_size_step
            print("best accuracy!\n")

        scheduler.step()
    print("best accuracy: {:.2f}%\n".format(100. * best / len(test_loader.dataset)))

    # Obtain the Current Time
    now = datetime.datetime.now()
    # Convert the Timestamp to a String
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")

    # Saving the model and training details if specified.
    if args.save_model:
        torch.save(best_dict, "fashionmnist_hardware_ep_program_{}_".format(best)+timestamp_str+".pt")
        shape_list = []
        for i in best_dict.keys():
            if "weight" in i:
                shape_list.append([k for k in best_dict[i].shape])
        # Additional details including architecture, epochs, and accuracy are written to a text file.
        with open("fashionmnist_hardware_ep_program_{}_".format(best)+timestamp_str+".txt", 'w') as f:
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
            f.write('program_noise: {:.1f}%\n'.format(params['p_noise']))
            f.write('sparsity: {}\n'.format(params['sparsity']))
            f.write('accuracy: {}/{} ({:.2f}%)\n'.format(best, len(test_loader.dataset), 100. * best / len(test_loader.dataset)))
            L = 0
            sum = 0
            for l in switch_count_list:
                program_count = int(l[-1].sum() - args.sparsity * l[-1].numel())
                f.write('layer{} average_program_counts: {:.0f}/({}/{})\n'.format(L, program_count / l[-1].numel(), program_count, l[-1].numel()))
                sum += program_count
                L += 1
            f.write('program_sum: {}\n'.format(sum))
            f.write('train epoch accuracy:\n')
            for i in acc_list:
                f.write('{:.4f}\n'.format(i / len(test_loader.dataset)))
            f.close()


if __name__ == '__main__':
    main()
