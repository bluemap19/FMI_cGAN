import argparse
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch, gc
from Dataloader_FMI_Gan import dataloader_fmi_mask_create
from model_fmi.netD_FMI import FMI_Discriminator
from model_fmi.netG_FMI import FMI_Generator
from math import sqrt
# import torch
from src_ele.pic_opeeration import cal_pic_generate_effect, show_Pic

gc.collect()
torch.cuda.empty_cache()


# def cal_pic_generate_effect(pic_org, pic_repair):
#     # print(pic_org.shape, pic_repair.shape)
#     # 计算PSNR：
#     PSNR = peak_signal_noise_ratio(pic_org, pic_repair)
#     # 计算SSIM
#     SSIM = structural_similarity(pic_org, pic_repair)
#     # 计算MSE 、 RMSE、 MAE、r2
#     mse = np.sum((pic_org - pic_repair) ** 2) / pic_org.size
#     rmse = sqrt(mse)
#     mae = np.sum(np.absolute(pic_org - pic_repair)) / pic_org.size
#     r2 = 1 - mse / np.var(pic_org)  # 均方误差/方差
#
#     return PSNR, SSIM, mse, rmse, mae, r2




if __name__ == '__main__':
    os.makedirs("images", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size_x", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--img_size_y", type=int, default=128, help="size of random mask")
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    # parser.add_argument("--dataset_path_str", type=str, default=r'D:\Data\target_stage1_small_big_mix',
    parser.add_argument("--dataset_path_str", type=str, default=r'D:\Data\ele_img_big_small_mix_GAN_test',
                        help="dataset path where to load")

    # parser.add_argument("--img_size_x", type=int, default=128, help="size of each image dimension")
    # parser.add_argument("--img_size_y", type=int, default=128, help="size of random mask")
    # parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    # parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    # parser.add_argument("--dataset_path_str", type=str, default=r'/root/autodl-tmp/data/target_stage1_small_big_mix',
    #                     help="dataset path where to load")


    parser.add_argument("--channels_in", type=int, default=2, help="number of image channels")
    parser.add_argument("--channels_out", type=int, default=2, help="number of image channels")

    parser.add_argument("--netG", type=str, default=r'D:\Data\pix2pix_exp\context_encoder_model_1\model_ele_gen_700.pth', help="netG path to load")
    parser.add_argument("--netD", type=str, default='', help="netD path to load")
    opt = parser.parse_args()

    # print(opt)

    cuda = True if torch.cuda.is_available() else False

    # Initialize generator and discriminator
    generator = FMI_Generator(channels_in=opt.channels_in, channels_out=opt.channels_out)

    if len(opt.netG)>1:
        if cuda:
            generator.load_state_dict(torch.load(opt.netG), strict=True)
        else:
            generator.load_state_dict(torch.load(opt.netG, map_location=torch.device('cpu')), strict=True)
    else:
        print('no available netG path:{}'.format(opt.netG))


    if cuda:
        generator.cuda()


    # if opt.netG != '':
    #     print('from model continue to train.........')
    #     generator.load_state_dict(torch.load(opt.netG), strict=True)
    # else:
    #     print('train a new model .......... ')
    # if opt.netD != '':
    #     discriminator.load_state_dict(torch.load(opt.netD), strict=True)

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

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    # def save_sample(imgs, masked, batches_done):
    #     gen_mask = generator(masked)
    #     # print(gen_mask.shape)
    #
    #     # Save sample
    #     imgs = torch.cat((imgs.data, masked.data, gen_mask.data), 1)
    #     imgs = imgs.reshape((imgs.shape[0]*imgs.shape[1], 1, imgs.shape[-2], imgs.shape[-1]))
    #     # print(imgs.shape)
    #
    #     save_image(imgs, "images/%d.png" % batches_done, nrow=6, padding=1, normalize=False)
    #
    #     model_path = "model_{}_{}.pth".format('ele_gen', batches_done)
    #     torch.save(generator.state_dict(), model_path)
    #     model_path = "model_{}_{}.pth".format('ele_dis', batches_done)
    #     torch.save(discriminator.state_dict(), model_path)


    # ----------
    #  Training
    # ----------
    dyna_M = []
    stat_M = []
    with torch.no_grad():
        for i, (imgs, masked) in enumerate(dataloader):
            # Configure input
            imgs = Variable(imgs.type(Tensor))
            masked = Variable(masked.type(Tensor))
            # print(imgs.shape, masked_imgs.shape, masked_parts.shape)

            # Generate a batch of images
            gen_parts = generator(masked).cpu().detach().numpy()
            imgs = imgs.cpu().detach().numpy()

            print(gen_parts.shape)          # torch.Size([3, 2, 128, 128])
            for j in range(gen_parts.shape[0]):
                # 动态图像的效果对比
                PSNR, SSIM, mse, rmse, mae, r2, Entropy_org, Entropy_vice, Con_org, Con_vice = cal_pic_generate_effect(imgs[j, 0, :, :], gen_parts[j, 0, :, :])
                dyna_M.append([PSNR, SSIM, mse, rmse, mae, r2, Entropy_org, Entropy_vice, Con_org, Con_vice])
                print('dyna - PSNR:{:.4f} SSIM:{:.4f} mse:{:.4f} rmse:{:.4f} mae:{:.4f} '
                      'r2:{:.4f} Entropy:{:.4f},{:.4f} Contrast:{:.4f},{:.4f}'.format(
                    PSNR, SSIM, mse, rmse, mae, r2, Entropy_org, Entropy_vice, Con_org, Con_vice))

                # 静态图像的效果对比
                PSNR, SSIM, mse, rmse, mae, r2, Entropy_org, Entropy_vice, Con_org, Con_vice = cal_pic_generate_effect(imgs[j, 1, :, :], gen_parts[j, 1, :, :])
                stat_M.append([PSNR, SSIM, mse, rmse, mae, r2, Entropy_org, Entropy_vice, Con_org, Con_vice])
                print('stat - PSNR:{:.4f} SSIM:{:.4f} mse:{:.4f} rmse:{:.4f} mae:{:.4f} '
                      'r2:{:.4f} Entropy:{:.4f},{:.4f} Contrast:{:.4f},{:.4f}'.format(
                    PSNR, SSIM, mse, rmse, mae, r2, Entropy_org, Entropy_vice, Con_org, Con_vice))

                show_Pic([1-imgs[j, 0, :, :], 1-gen_parts[j, 0, :, :], 1-imgs[j, 1, :, :], 1-gen_parts[j, 1, :, :]], pic_order='22')

        dyna_M = np.array(dyna_M)
        stat_M = np.array(stat_M)
        M_all = []
        for i in range(dyna_M.shape[1]):
            M_Temp = np.mean(dyna_M[:, i])
            M_all.append(M_Temp)
            M_Temp = np.mean(stat_M[:, i])
            M_all.append(M_Temp)

        print(M_all)


