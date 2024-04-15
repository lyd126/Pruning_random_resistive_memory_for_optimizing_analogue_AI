import os
import argparse
import time
import datetime
import math
import numpy as np
import copy
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd
import torch.backends.cudnn as cudnn

from lib.losses.loss import *
from lib.common import *
from collections import OrderedDict
from lib.metrics import Evaluate
from lib.logger import Logger, Print_Logger
import datetime


args = None
init_bin = True
limit_size_start = None
limit_size_stop = None
limit_size_step = None
limit_size = None
noise = 0.2


def inject_noise(w):
    out = w.clone()
    sigma = noise / 100 * out.abs()
    out_n = torch.normal(out, sigma)
    p2s = out + 2 * sigma
    n2s = out - 2 * sigma
    out_n = torch.where(out_n > p2s, p2s, out_n)
    out_n = torch.where(out_n < n2s, n2s, out_n)

    return out_n


def quantization(m, bit, after_relu=False):
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
    def forward(ctx, scores, k, mask_old, score_old, limit_size):
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
        return g, None, None, None, None


class MaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        self.hardware_weight_one = self.weight.data.cuda()
        self.hardware_weight_zero = self.weight.data.cuda()

        # NOTE: initialize the weights like this.
        fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
        # scale:
        fan = fan * (1 - 0.5)
        gain = nn.init.calculate_gain("relu")
        self.std = gain / math.sqrt(fan)
        self.weight.data = self.weight.data.sign() * self.std
        self.scale = self.hardware_weight_one.mean() / self.std
        self.sign = self.weight.data.sign().cuda()

        self.subnet_old = torch.ones_like(self.weight.data).cuda()
        self.score_old = self.scores.data.clone()

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

    def forward(self, x, sparsity, limit_size, switch_count=None):
        if switch_count is not None:
            subnet = GetSubnet.apply(self.scores.abs(), sparsity, self.subnet_old, self.score_old.abs(), limit_size)
            self.subnet_old = subnet.data
            self.score_old = self.scores.data.clone()
            switch_count[-1] += torch.where(subnet != switch_count[0], 1.0, 0.0).cuda()
            w = self.sign * (self.hardware_weight_one * subnet + self.hardware_weight_zero * (1 - subnet))
            x = F.conv2d(
                x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
            switch_count[0] = subnet.data.clone()
            return x / self.scale, switch_count
        else:
            subnet = GetSubnet.apply(self.scores.abs(), sparsity, self.subnet_old, self.score_old.abs(), limit_size)
            w = self.sign * (self.hardware_weight_one * subnet + self.hardware_weight_zero * (1 - subnet))
            x = F.conv2d(
                x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
            return x / self.scale


class MaskConvTranspose(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        self.hardware_weight_one = self.weight.data.cuda()
        self.hardware_weight_zero = self.weight.data.cuda()

        # NOTE: initialize the weights like this.
        fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
        # scale:
        fan = fan * (1 - 0.5)
        gain = nn.init.calculate_gain("relu")
        self.std = gain / math.sqrt(fan)
        self.weight.data = self.weight.data.sign() * self.std
        self.scale = self.hardware_weight_one.mean() / self.std
        self.sign = self.weight.data.sign().cuda()

        self.subnet_old = torch.ones_like(self.weight.data).cuda()
        self.score_old = self.scores.data.clone()

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

    def forward(self, x, sparsity, limit_size, switch_count=None):
        if switch_count is not None:
            subnet = GetSubnet.apply(self.scores.abs(), sparsity, self.subnet_old, self.score_old.abs(), limit_size)
            self.subnet_old = subnet.data
            self.score_old = self.scores.data.clone()
            switch_count[-1] += torch.where(subnet != switch_count[0], 1.0, 0.0).cuda()
            w = self.sign * (self.hardware_weight_one * subnet + self.hardware_weight_zero * (1 - subnet))
            num_spatial_dims = 2
            output_padding = self._output_padding(x, None, self.stride, self.padding, self.kernel_size,
                                                  num_spatial_dims, self.dilation)
            x = F.conv_transpose2d(x, w, self.bias, self.stride, self.padding, output_padding, self.groups,
                                   self.dilation)
            switch_count[0] = subnet.data.clone()
            return x / self.scale, switch_count
        else:
            self.subnet_old = torch.ones_like(self.weight.data).cuda()
            subnet = GetSubnet.apply(self.scores.abs(), sparsity, self.subnet_old, self.score_old.abs(), limit_size)
            w = self.sign * (self.hardware_weight_one * subnet + self.hardware_weight_zero * (1 - subnet))
            num_spatial_dims = 2
            output_padding = self._output_padding(x, None, self.stride, self.padding, self.kernel_size,
                                                  num_spatial_dims, self.dilation)
            x = F.conv_transpose2d(x, w, self.bias, self.stride, self.padding, output_padding, self.groups,
                                      self.dilation)
            return x / self.scale


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__()
        self.conv1 = MaskConv(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = MaskConv(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, sparsity, limit_size, qb=None, switch_count_list=None):
        if qb:
            try:
                x.data = quantization(x, qb)
            except:
                pass
        if switch_count_list:
            x, switch_count_list[0] = self.conv1(x, sparsity, limit_size, switch_count_list[0])
            x = self.bn1(x)
            x = self.relu(x)
            if qb:
                try:
                    x.data = quantization(x, qb)
                except:
                    pass
            x, switch_count_list[1] = self.conv2(x, sparsity, limit_size, switch_count_list[1])
            x = self.bn2(x)
            x = self.relu(x)
            return x, switch_count_list
        else:
            x = self.conv1(x, sparsity, limit_size)
            x = self.bn1(x)
            x = self.relu(x)
            if qb:
                try:
                    x.data = quantization(x, qb)
                except:
                    pass
            x = self.conv2(x, sparsity, limit_size)
            x = self.bn2(x)
            x = self.relu(x)
            return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.MaxPool = nn.MaxPool2d(2, stride=2)
        self.DownConv = DoubleConv(in_channels, out_channels)

    def forward(self, x, sparsity, limit_size, qb=None, switch_count_list=None):
        x = self.MaxPool(x)
        if switch_count_list:
            x, switch_count_list = self.DownConv(x, sparsity, limit_size, qb, switch_count_list)
            return x, switch_count_list
        else:
            x = self.DownConv(x, sparsity, limit_size, qb)
            return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = MaskConvTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2, sparsity, limit_size, qb=None, switch_count_list=None):
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        if switch_count_list:
            x, switch_count_list = self.conv(x, sparsity, limit_size, qb, switch_count_list)
            return x, switch_count_list
        else:
            x = self.conv(x, sparsity, limit_size, qb)
            return x


class OutConv(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__()
        self.out_conv = MaskConv(in_channels, num_classes, kernel_size=1, bias=False)

    def forward(self, x, sparsity, limit_size, qb=None, switch_count_list=None):
        if qb:
            try:
                x.data = quantization(x, qb)
            except:
                pass
        if switch_count_list:
            x, switch_count_list = self.out_conv(x, sparsity, limit_size, switch_count_list)
            return x, switch_count_list
        else:
            x = self.out_conv(x, sparsity, limit_size)
            return x


class UNet_ep(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_ep, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x, sparsity, limit_size, qb=6, switch_count_list=None):
        if switch_count_list:
            x1, switch_count_list[: 2] = self.in_conv(x, sparsity, limit_size, qb, switch_count_list[: 2])
            x2, switch_count_list[2: 4] = self.down1(x1, sparsity, limit_size, qb, switch_count_list[2: 4])
            x3, switch_count_list[4: 6] = self.down2(x2, sparsity, limit_size, qb, switch_count_list[4: 6])
            x4, switch_count_list[6: 8] = self.down3(x3, sparsity, limit_size, qb, switch_count_list[6: 8])
            x5, switch_count_list[8: 10] = self.down4(x4, sparsity, limit_size, qb, switch_count_list[8: 10])
            x, switch_count_list[10: 12] = self.up1(x5, x4, sparsity, limit_size, qb, switch_count_list[10: 12])
            x, switch_count_list[12: 14] = self.up2(x, x3, sparsity, limit_size, qb, switch_count_list[12: 14])
            x, switch_count_list[14: 16] = self.up3(x, x2, sparsity, limit_size, qb, switch_count_list[14: 16])
            x, switch_count_list[16: 18] = self.up4(x, x1, sparsity, limit_size, qb, switch_count_list[16: 18])
            out, switch_count_list[18] = self.out_conv(x, sparsity, limit_size, qb, switch_count_list[18])

            return F.softmax(out, dim=1), switch_count_list
        else:
            x1 = self.in_conv(x, sparsity=sparsity, limit_size=limit_size, qb=qb)
            x2 = self.down1(x1, sparsity=sparsity, limit_size=limit_size, qb=qb)
            x3 = self.down2(x2, sparsity=sparsity, limit_size=limit_size, qb=qb)
            x4 = self.down3(x3, sparsity=sparsity, limit_size=limit_size, qb=qb)
            x5 = self.down4(x4, sparsity=sparsity, limit_size=limit_size, qb=qb)
            x = self.up1(x5, x4, sparsity=sparsity, limit_size=limit_size, qb=qb)
            x = self.up2(x, x3, sparsity=sparsity, limit_size=limit_size, qb=qb)
            x = self.up3(x, x2, sparsity=sparsity, limit_size=limit_size, qb=qb)
            x = self.up4(x, x1, sparsity=sparsity, limit_size=limit_size, qb=qb)
            out = self.out_conv(x, sparsity=sparsity, limit_size=limit_size, qb=qb)

            return F.softmax(out, dim=1)


def train(model, switch_count_list, qb, sparsity, limit_size, device, train_loader, optimizer, criterion=CrossEntropyLoss2d()):
    model.train()
    train_loss = AverageMeter()

    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs, switch_count_list = model(data, sparsity, limit_size, qb, switch_count_list)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), data.size(0))
    log = OrderedDict([('train_loss', train_loss.avg)])

    return switch_count_list, log


def val(model, qb, sparsity, limit_size, device, val_loader, criterion=CrossEntropyLoss2d()):
    model.eval()
    val_loss = AverageMeter()
    evaluater = Evaluate()
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            data, target = data.to(device), target.to(device)
            outputs = model(data, sparsity, limit_size, qb=qb)
            loss = criterion(outputs, target)
            val_loss.update(loss.item(), data.size(0))
            outputs = outputs.data.cpu().numpy()
            target = target.data.cpu().numpy()
            evaluater.add_batch(target, outputs[:, 1])
        log = OrderedDict([('val_loss', val_loss.avg),
                           ('val_acc', evaluater.confusion_matrix()[1]),
                           ('val_f1', evaluater.f1_score()),
                           ('val_auc_roc', evaluater.auc_roc()),
                           ('val_auc_pr', evaluater.auc_pr())])

        return log


def main():
    global args
    global limit_size_start
    global limit_size_stop
    global limit_size_step
    global limit_size

    # Training settings
    parser = argparse.ArgumentParser(description="Pytorch Unet with Hardware TO")

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--q_bit', type=int, default=6, metavar='N',
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

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the best Model')
    parser.add_argument('--save-path', type=str, default='save', help='Path for saving the best Model')
    parser.add_argument('--data', type=str, default='./', help='Location to store data')
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='how sparse is each layer')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    cudnn.benchmark = True

    log = Logger(args.save_path)
    sys.stdout = Print_Logger(os.path.join(args.save_path, 'unet_hardware_ep_train_log.txt'))

    train_loader = torch.load('UNet_train_loader.pt')
    test_loader = torch.load('UNet_test_loader.pt')

    model = UNet_ep(in_channels=1, base_c=32)
    model.to(device)
    print(model)

    one_weight_dict = torch.load('UNet_RRAM_hardware_one_weight.pt')
    zero_weight_dict = torch.load('UNet_RRAM_hardware_zero_weight.pt')


    for name1, module1 in model.named_children():
        if ('up' in name1) or ('down' in name1):
            for name2, module2 in module1.named_children():
                if 'onv' not in name2:
                    pass
                else:
                    for name3, layer in module2.named_children():
                        if 'onv' not in name3:
                            pass
                        else:
                            for key in one_weight_dict.keys():
                                if key.split('.weight')[0] == name1 + '.' + name2 + '.' + name3:
                                    layer.hardware_weight_one = torch.from_numpy(
                                        one_weight_dict[key].copy()).abs().float().cuda()
                                    layer.hardware_weight_zero = torch.from_numpy(
                                        zero_weight_dict[key].copy()).abs().float().cuda()
                                    layer.scale = layer.hardware_weight_one.mean() / layer.std
                                    break
                                else:
                                    pass
        else:
            for name2, layer in module1.named_children():
                if 'onv' not in name2:
                    pass
                else:
                    for key in one_weight_dict.keys():
                        if key.split('.weight')[0] == name1 + '.' + name2:
                            layer.hardware_weight_one = torch.from_numpy(
                                one_weight_dict[key].copy()).abs().float().cuda()
                            layer.hardware_weight_zero = torch.from_numpy(
                                zero_weight_dict[key].copy()).abs().float().cuda()
                            layer.scale = layer.hardware_weight_one.mean() / layer.std
                            break
                        else:
                            pass

    Reset = 0
    Set = 0

    switch_count_list = []
    for key in one_weight_dict.keys():
        switch_count_list.append([
            torch.ones(one_weight_dict[key].shape).cuda(),
            torch.zeros(one_weight_dict[key].shape).cuda()
        ])

    one_weight_dict = 0
    zero_weight_dict = 0

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, momentum=args.momentum, weight_decay=args.wd
    )

    # optimizer = optim.Adam(
    #     [p for p in model.parameters() if p.requires_grad],
    #     lr=args.lr, weight_decay=args.wd
    # )

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    limit_size_start = 0.001
    limit_size_stop = 0.0008
    limit_size_step = 5
    limit_size = limit_size_start

    best = {'epoch': 0, 'AUC_roc': 0.5}
    auc_roc_list = []

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        switch_count_list, train_log = train(model, switch_count_list, args.q_bit, args.sparsity, limit_size, device, train_loader, optimizer)
        val_log = val(model, args.q_bit, args.sparsity, limit_size, device, test_loader)
        log.update(epoch, train_log, val_log)
        lr_scheduler.step()

        auc_roc_list.append([val_log['val_auc_roc'], val_log['val_auc_pr']])

        if val_log['val_auc_roc'] > best['AUC_roc']:
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'limit_size': limit_size}
            best_dict = copy.deepcopy(state)
            best['epoch'] = epoch
            best['AUC_roc'] = val_log['val_auc_roc']
            print('Best AUC_roc!')
    print('Best performance at Epoch: {} | AUC_roc: {:.3f}'.format(best['epoch'], best['AUC_roc']))

    if args.save_model:
        print('\033[0;33mSaving best model!\033[0m')
        torch.save(best_dict, './save/unet_hardware_ep_{:.4f}.pt'.format(best['AUC_roc']))
        shape_list = []
        for i in best_dict['net'].keys():
            if "weight" in i:
                shape_list.append([k for k in best_dict['net'][i].shape])
        with open("./save/unet_hardware_ep_{:.4f}.txt".format(best['AUC_roc']), 'w') as f:
            f.write('Model: Hardware EP\n')
            f.write('arch:\n')
            for s in shape_list:
                arch = str(s[0])
                for t in s[1:]:
                    arch = arch + '_{}'.format(t)
                f.write('\t' + arch + '\n')
            f.write('number of params (M): %.2f\n' % (n_parameters / 1.e6))
            f.write('training_epochs: {}\n'.format(args.epochs))
            f.write('batch_size: {}\n'.format(args.batch_size))
            f.write('input_bit: {}\n'.format(args.q_bit))
            f.write('sparsity: {}\n'.format(args.sparsity))
            f.write('limit_size_start: {}\n'.format(limit_size_start))
            f.write('limit_size_stop: {}\n'.format(limit_size_stop))
            f.write('limit_size_step: {}\n'.format(limit_size_step))
            L = 0
            SUM = 0
            for l in switch_count_list:
                program_count = int(l[-1].sum() - args.sparsity * l[-1].numel())
                f.write('layer{} average_program_counts: {:.0f}/({}/{})\n'.format(L, program_count / l[-1].numel(),
                                                                                  program_count, l[-1].numel()))
                SUM += program_count
                L += 1
            f.write('program_sum: {}\n'.format(SUM))
            f.write('best AUC_roc: {:.3f}\n'.format(best['AUC_roc']))
            f.write('train epoch AUC_roc and AUC_pr:\n')
            for i in auc_roc_list:
                f.write('{:.3f}, {:.3f}\n'.format(i[0], i[1]))
            f.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


if __name__ == '__main__':
    if not os.path.exists("./save"):
        os.mkdir("./save")
    main()