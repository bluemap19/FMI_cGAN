import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from datasets import ImageDataset_FMI
from models import GeneratorUNet, Discriminator, weights_init_normal


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=106, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=126, help="number of epochs of training")
# parser.add_argument("--dataset_path", type=str, default="FMI_GAN", help="name of the dataset")


parser.add_argument("--img_size_x", type=int, default=256, help="size of image height")
parser.add_argument("--img_size_y", type=int, default=256, help="size of image width")
parser.add_argument("--channels_in", type=int, default=2, help="number of image channels")
parser.add_argument("--channels_out", type=int, default=2, help="number of image channels")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
# parser.add_argument("--dataset_path", type=str, default=r"/root/autodl-tmp/data/target_stage1_small_big_mix",
parser.add_argument("--dataset_path", type=str, default=r"/root/autodl-tmp/data/GAN_Fracture_layer",
                    help="path of the dataset")


parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between model_fmi checkpoints")
parser.add_argument("--dataset_name", type=str, default='pic2pic', help="folder to save model")
parser.add_argument("--netG", type=str, default=r'/root/autodl-tmp/FMI_GAN/pix2pix/saved_models/pic2pic/generator_104.pth', help="path model Gen")
parser.add_argument("--netD", type=str, default='/root/autodl-tmp/FMI_GAN/pix2pix/saved_models/pic2pic/discriminator_104.pth', help="path model Discrimi")
opt = parser.parse_args()
print(opt)


os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_size_x // 2 ** 4, opt.img_size_y // 2 ** 4)


# Initialize generator and discriminator
generator = GeneratorUNet(in_channels=opt.channels_in, out_channels=opt.channels_out)
discriminator = Discriminator(in_channels=opt.channels_out)


if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

# if opt.epoch != 0:
#     # Load pretrained models
#     generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
#     discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
# else:
#     # Initialize weights
#     generator.apply(weights_init_normal)
#     discriminator.apply(weights_init_normal)
if opt.netG != '':
    print('from model continue to train.........')
    generator.load_state_dict(torch.load(opt.netG), strict=True)
else:
    print('train a new model .......... ')
    generator.apply(weights_init_normal)
if opt.netD != '':
    discriminator.load_state_dict(torch.load(opt.netD), strict=True)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))



dataloader = ImageDataset_FMI(opt.dataset_path, x_l=opt.img_size_x, y_l=opt.img_size_x)
dataloader = DataLoader(dataloader, shuffle=False, batch_size=opt.batch_size, drop_last=True, pin_memory=True,
                        num_workers=opt.n_cpu)


# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



def sample_images(imgs, masked, batches_done):
    gen_mask = generator(masked)

    # Save sample
    imgs = torch.cat((imgs.data, masked.data, gen_mask.data), 1)
    imgs = imgs.reshape((imgs.shape[0] * imgs.shape[1], 1, imgs.shape[-2], imgs.shape[-1]))
    # print(imgs.shape)

    save_image(imgs, "images/%d.png" % batches_done, nrow=6, padding=1, normalize=False)
    # model_path = "model_{}_{}.pth".format('ele_gen', batches_done)
    # torch.save(generator.state_dict(), model_path)
    # model_path = "model_{}_{}.pth".format('ele_dis', batches_done)
    # torch.save(discriminator.state_dict(), model_path)

# def sample_images(batches_done):
#     """Saves a generated sample from the validation set"""
#     imgs = next(iter(val_dataloader))
#     real_A = Variable(imgs["B"].type(Tensor))
#     real_B = Variable(imgs["A"].type(Tensor))
#     fake_B = generator(real_A)
#     img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
#     save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)


# ----------
#  Training
# ----------

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Model inputs
        real_A = Variable(batch["B"].type(Tensor))
        real_B = Variable(batch["A"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()


        # GAN loss
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)


        # Total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(real_B, real_A, batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model_fmi checkpoints
        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))
