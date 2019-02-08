import argparse

import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


def get_args():
    """
    Parses input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--content-path", type=str, default="assets/cnt_img.jpg",
                        help="Path for a content image")
    parser.add_argument("-s", "--style-path", type=str, default="assets/st_img.jpg",
                        help="Path for a style image")
    parser.add_argument("-r", "--size", type=int, default=240,
                        help="Resize images to this size")
    parser.add_argument("-i", "--n-iterations", type=int, default=1000,
                        help="Number of iterations")
    parser.add_argument("-o", "--output-path", type=str, default="assets/output.jpg",
                        help="Path to output file")
    parser.add_argument("-l", "--show-loss", type=bool, default=False,
                        help="Decides if loss is printed every 100th iteration")
    return vars(parser.parse_args())


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

normalize = transforms.Compose([
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

denormalize = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=1/imagenet_std),
    transforms.Normalize(mean=imagenet_mean*(-1), std=[1., 1., 1.]),
])


def imgs2tensors(cnt_img_path, st_img_path, size):
    """
    Converts content image and style images from a given paths to normalized torch tensors of defined size
    """

    raw_cnt_img = Image.open(cnt_img_path)
    raw_st_img = Image.open(st_img_path)

    cnt_tfms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    cnt_img = cnt_tfms(raw_cnt_img).unsqueeze_(0)

    st_tfms = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop((cnt_img.size()[2], cnt_img.size()[3])),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    st_img = st_tfms(raw_st_img).unsqueeze_(0)

    if torch.cuda.is_available():
        cnt_img_v = cnt_img.cuda()
        st_img_v = st_img.cuda()

    return cnt_img, st_img


def get_optimizer_img(shape):
    """
    Creates an optimizer image from random noise
    """
    opt_img = np.random.uniform(0, 1, size=shape[1:]).astype(np.float32)
    opt_img = scipy.ndimage.filters.median_filter(opt_img, [8, 8, 1])
    opt_img = torch.Tensor(opt_img)
    opt_img = normalize(opt_img).unsqueeze(0)
    return opt_img


class FeaturesHook:
    """
    Class that allows to save weights from given layers
    """
    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self):
        self.hook.remove()


def gram(input):
    b, c, h, w = input.size()
    x = input.view(b*c, -1)
    return torch.mm(x, x.t())/input.numel()*1e6


def gram_mse_loss(input, target):
    return F.mse_loss(gram(input), gram(target))


def save_img(tensor, path):
    tensor = denormalize(tensor)
    save_image(tensor, path)