import argparse
import os
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch, gc
from Dataloader_FMI_Gan import dataloader_fmi_mask_create
from model_fmi.netD_FMI import FMI_Discriminator
from model_fmi.netG_FMI import FMI_Generator

# import torch

gc.collect()
torch.cuda.empty_cache()


if __name__ == '__main__':
    os.makedirs("images", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

    # parser.add_argument("--img_size_x", type=int, default=32, help="size of each image dimension")
    # parser.add_argument("--img_size_y", type=int, default=32, help="size of random mask")
    # parser.add_argument("--batch_size", type=int, default=3, help="size of the batches")
    # parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    # parser.add_argument("--dataset_path_str", type=str, default=r'D:\Data\target_stage1_small_big_mix',
    #                     help="dataset path where to load")

    parser.add_argument("--img_size_x", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--img_size_y", type=int, default=128, help="size of random mask")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    # parser.add_argument("--dataset_path_str", type=str, default=r'/root/autodl-tmp/data/target_stage1_small_big_mix',
    parser.add_argument("--dataset_path_str", type=str, default=r'/root/autodl-tmp/data/GAN_Fracture_layer',
                        help="dataset path where to load")

    parser.add_argument("--channels_in", type=int, default=2, help="number of image channels")
    parser.add_argument("--channels_out", type=int, default=2, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling, when to save train sample effect")
    # parser.add_argument("--netG", type=str, default='model_ele_gen_1.pth', help="netG path")
    # parser.add_argument("--netD", type=str, default='model_ele_dis_1.pth', help="netD path")
    parser.add_argument("--netG", type=str, default=r'/root/autodl-tmp/FMI_GAN/model_ele_gen_2500.pth', help="netG path to load")
    parser.add_argument("--netD", type=str, default=r'/root/autodl-tmp/FMI_GAN/model_ele_dis_2500.pth', help="netD path to load")
    opt = parser.parse_args()

    # print(opt)

    cuda = True if torch.cuda.is_available() else False


    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)


    # Loss function
    adversarial_loss = torch.nn.MSELoss()
    pixelwise_loss = torch.nn.L1Loss()



    # Initialize generator and discriminator
    generator = FMI_Generator(channels_in=opt.channels_in, channels_out=opt.channels_out)
    discriminator = FMI_Discriminator(channels=opt.channels_out)


    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        pixelwise_loss.cuda()


    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)


    # props = torch.cuda.get_device_properties(0)
    # total_memory = props.total_memory // 1024 ** 2  # 转换为MB
    # free_memory = torch.cuda.memory_allocated(0) // 1024 ** 2  # 转换为MB
    # print("initialize model: gen and discriminator 总内存: {}MB, 可用内存: {}MB".format(total_memory, free_memory))

    if opt.netG != '':
        print('from model continue to train.........')
        generator.load_state_dict(torch.load(opt.netG), strict=True)
    else:
        print('train a new model .......... ')
    if opt.netD != '':
        discriminator.load_state_dict(torch.load(opt.netD), strict=True)

    # Dataset loader
    # dataloader = DataLoader(
    #     ImageDataset(r"D:\XL_Download\Streetview\0", transforms_=transforms_),
    #     batch_size=opt.batch_size,
    #     shuffle=True,
    #     num_workers=opt.n_cpu,
    # )

    print('loading data........')
    dataloader = dataloader_fmi_mask_create(path=opt.dataset_path_str, x_l=opt.img_size_x, y_l=opt.img_size_x)                            # , transform=repair_ele_dataloader.dataset_collate_ele_repair
    dataloader = DataLoader(dataloader, shuffle=False, batch_size=opt.batch_size, drop_last=True, pin_memory=True, num_workers=opt.n_cpu)


    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    def save_sample(imgs, masked, batches_done):
        # imgs, masked_samples, i = next(iter(dataloader))
        # print(imgs.shape, masked_samples.shape, i.shape)

        # imgs = torch.reshape(imgs, [imgs.shape[0], 1, imgs.shape[1], imgs.shape[2]])
        # imgs = Variable(imgs.type(Tensor))
        # masked_samples = Variable(masked_samples.type(Tensor))
        # Generate inpainted image
        gen_mask = generator(masked)
        # print(gen_mask.shape)

        # Save sample
        imgs = torch.cat((imgs.data, masked.data, gen_mask.data), 1)
        imgs = imgs.reshape((imgs.shape[0]*imgs.shape[1], 1, imgs.shape[-2], imgs.shape[-1]))
        # print(imgs.shape)

        save_image(imgs, "images/%d.png" % batches_done, nrow=6, padding=1, normalize=False)

        model_path = "model_{}_{}.pth".format('ele_gen', batches_done)
        torch.save(generator.state_dict(), model_path)
        model_path = "model_{}_{}.pth".format('ele_dis', batches_done)
        torch.save(discriminator.state_dict(), model_path)


    # ----------
    #  Training
    # ----------

    for epoch in range(opt.n_epochs):
        for i, (imgs, masked) in enumerate(dataloader):
            # imgs = torch.reshape(imgs, [imgs.shape[0], 2, imgs.shape[-2], imgs.shape[-1]])
            # masked_parts = torch.reshape(masked_parts, [masked_parts.shape[0], 2, masked_parts.shape[-2], masked_parts.shape[-1]])
            # print('input shape:{}, output shape:{}'.format(imgs.shape, masked.shape))

            # Adversarial ground truths
            # 为了实现自动求导功能，引入了Variable
            valid = Variable(Tensor(opt.batch_size, 1, opt.img_size_x//4, opt.img_size_y//4).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(opt.batch_size, 1, opt.img_size_x//4, opt.img_size_y//4).fill_(0.0), requires_grad=False)
            if i==0:
                props = torch.cuda.get_device_properties(0)
                total_memory = props.total_memory // 1024 ** 2  # 转换为MB
                free_memory = torch.cuda.memory_allocated(0) // 1024 ** 2  # 转换为MB
                print("valid, fake:总内存: {}MB, 可用内存: {}MB".format(total_memory, free_memory))

            # Configure input
            imgs = Variable(imgs.type(Tensor))
            masked = Variable(masked.type(Tensor))
            # print(imgs.shape, masked_imgs.shape, masked_parts.shape)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_parts = generator(masked)
            if i==0:
                props = torch.cuda.get_device_properties(0)
                total_memory = props.total_memory // 1024 ** 2  # 转换为MB
                free_memory = torch.cuda.memory_allocated(0) // 1024 ** 2  # 转换为MB
                print("gen model:总内存: {}MB, 可用内存: {}MB".format(total_memory, free_memory))

            # dis_part = discriminator(gen_parts)
            # print('dis_part shape is:{}'.format(dis_part.shape))

            # Adversarial and pixelwise loss
            g_adv = adversarial_loss(discriminator(gen_parts), valid)
            g_pixel = pixelwise_loss(gen_parts, imgs)

            # Total loss
            g_loss = 0.001 * g_adv + 0.999 * g_pixel

            g_loss.backward()
            optimizer_G.step()


            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_parts.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

            if i%200 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_adv.item(), g_pixel.item())
                )

            # Generate sample at sample interval
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_sample(imgs, masked, batches_done)