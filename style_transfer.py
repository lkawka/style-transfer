from utils import *

from pathlib import Path
import time

import torch
import torch.nn.functional as F
import torchvision.models as models


if __name__ == "__main__":
    args = get_args()

    content_img_path = Path(args["content_path"])
    style_img_path = Path(args["style_path"])
    size = args["size"]
    n_iterations = args["n_iterations"]
    output_path = args["output_path"]
    print_loss = args["show_loss"]

    # Prepare content and style images and make sure gradients are not computed
    cnt_img, st_img = imgs2tensors(content_img_path, style_img_path, size)
    cnt_img.requires_grad_(False)
    st_img.requires_grad_(False)

    save_img(cnt_img.detach().squeeze().cpu(), "cnt_img.jpg")
    save_img(st_img.detach().squeeze().cpu(), "st_img.jpg")
    exit()

    # blocks = [i for i, l in enumerate(list(models.vgg16_bn(pretrained=True).children())[0]) if isinstance(l, torch.nn.MaxPool2d)]
    blocks = [6, 13, 23, 33, 43]

    # Get pretrained VGG-16 model
    model = list(models.vgg16_bn(pretrained=True).children())[0]
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # Get image for optimization
    opt_img = get_optimizer_img(cnt_img.shape)
    opt_img.requires_grad_(True)

    # If possible move everything to GPU
    if torch.cuda.is_available():
        st_img.cuda()
        cnt_img.cuda()
        model.cuda()
        opt_img.cuda()

    hooks = [FeaturesHook(list(model.children())[idx]) for idx in blocks]

    model(cnt_img)
    target_cnt = [hook.features.clone() for hook in hooks]

    model(st_img)
    target_st = [hook.features.clone() for hook in hooks]

    optimizer = torch.optim.LBFGS([opt_img])

    # Regularization term for content loss
    alpha = 1e7

    # Loss function
    def gatys_loss(x):
        model(x)
        outs = [out.features for out in hooks]

        st_loss = sum([gram_mse_loss(o, s) for o, s in zip(outs, target_st)])
        cnt_loss = F.mse_loss(outs[3], target_cnt[3]) * alpha

        return st_loss + cnt_loss

    # Closure for LBFGS optimizer
    def closure():
        global itr, show_itr, print_loss
        optimizer.zero_grad()
        loss = gatys_loss(opt_img)
        loss.backward()
        itr += 1
        if print_loss and itr%show_itr == 0:
            print(f"Iteration: {itr}, loss: {loss}")
        return loss

    start_time = time.time()
    itr = 0
    show_itr = 100
    print("Transfer started")
    # Training
    while itr < n_iterations:
        optimizer.step(closure)
    print(f"Transfer ended. Total time: {int(time.time()-start_time)}s")

    save_img(opt_img.detach().squeeze(0).cpu(), output_path)