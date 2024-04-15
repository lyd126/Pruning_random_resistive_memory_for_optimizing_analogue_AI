import random
from os.path import join
from lib.extract_patches import get_data_train
from lib.losses.loss import *
from lib.visualize import group_images, save_img
from lib.common import *
from lib.dataset import TrainDataset
from torch.utils.data import DataLoader
from collections import OrderedDict
from lib.metrics import Evaluate
from lib.visualize import group_images, save_img
from lib.extract_patches import get_data_train
from lib.datasetV2 import data_preprocess, create_patch_idx, TrainDatasetV2
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def plot_weight(x, name, out_dir=None, aspect=1):
    plt.close("all")
    title_size = 44
    font_size = 38
    font = {'family': 'Times New Roman',
            'color': 'black',
            'weight': 'heavy',
            'size': font_size,
            }
    if out_dir is None:
        out_dir = 'C:/Users\ly\Desktop\pruning\pic\特征提取/unet/'

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'heavy'
    if x.shape[0] == 3:
        x = x.swapaxes(0, 2)
        x = x.swapaxes(0, 1)
        plt.imshow(x, interpolation='nearest', aspect=aspect)
    elif x.shape[0] == 1:
        plt.imshow(x[0], interpolation='nearest', cmap='inferno', aspect=aspect)
    elif x.shape[-1] == 1:
        plt.imshow(x[:, :, 0], interpolation='nearest', cmap='inferno', aspect=aspect)
    else:
        pass
    plt.title(name, pad=20, fontdict=font, fontsize=title_size)
    plt.xticks([], [])
    plt.yticks([], [])
    # cb1 = plt.colorbar(fraction=0.046, pad=0.04)
    # cb1.ax.tick_params(labelsize=18)
    # cb1.ax.yaxis.set_major_locator(MultipleLocator(0.5))
    # plt.show()
    fig = plt.gcf()
    fig.set_size_inches(19.20, 10.80)
    fig.savefig(out_dir + name + '.pdf', dpi=600, transparent=True)
    plt.close("all")

# ========================get dataloader==============================

def get_dataloader(args):
    """
    该函数加载数据集所有图像到内存，并创建训练样本提取位置的索引，所以占用内存量较少，
    测试结果表明，相比于上述原始的get_dataloader方法并不会降低训练效率
    """
    imgs_train, masks_train, fovs_train = data_preprocess(data_path_list=args.train_data_path_list)

    patches_idx = create_patch_idx(fovs_train, args)

    train_idx, val_idx = np.vsplit(patches_idx, (int(np.floor((1 - args.val_ratio) * patches_idx.shape[0])),))

    train_set = TrainDatasetV2(imgs_train, masks_train, fovs_train, train_idx, mode="train", args=args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)

    val_set = TrainDatasetV2(imgs_train, masks_train, fovs_train, val_idx, mode="val", args=args)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Save some samples of feeding to the neural network
    if args.sample_visualization:
        visual_set = TrainDatasetV2(imgs_train, masks_train, fovs_train, val_idx, mode="val", args=args)
        visual_loader = DataLoader(visual_set, batch_size=1, shuffle=True, num_workers=0)
        N_sample = 16
        visual_imgs = np.empty((N_sample, 1, args.train_patch_height, args.train_patch_width))
        visual_masks = np.empty((N_sample, 1, args.train_patch_height, args.train_patch_width))

        for i, (img, mask) in tqdm(enumerate(visual_loader)):
            visual_imgs[i] = np.squeeze(img.numpy(), axis=0)
            visual_masks[i, 0] = np.squeeze(mask.numpy(), axis=0)
            if i >= N_sample - 1:
                break
        save_img(group_images((visual_imgs[0:N_sample, :, :, :] * 255).astype(np.uint8), 4),
                 "D:\postphd\pruning/rram_online_training/Unet_new/save\experiment/sample_input_imgs.png")
        save_img(group_images((visual_masks[0:N_sample, :, :, :] * 255).astype(np.uint8), 10),
                 "D:\postphd\pruning/rram_online_training/Unet_new/save\experiment/sample_input_masks.png")
    return train_loader, val_loader


