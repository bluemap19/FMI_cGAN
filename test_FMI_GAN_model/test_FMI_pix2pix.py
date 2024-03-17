import argparse
import numpy as np
import time
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from pix2pix.datasets import ImageDataset_FMI
from pix2pix.models import GeneratorUNet
from src_ele.pic_opeeration import show_Pic, cal_pic_generate_effect

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    # parser.add_argument("--dataset_path", type=str, default="FMI_GAN", help="name of the dataset")


    parser.add_argument("--img_size_x", type=int, default=256, help="size of image height")
    parser.add_argument("--img_size_y", type=int, default=256, help="size of image width")
    parser.add_argument("--channels_in", type=int, default=2, help="number of image channels")
    parser.add_argument("--channels_out", type=int, default=2, help="number of image channels")

    # parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
    # parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    # parser.add_argument("--dataset_path", type=str, default=r"/root/autodl-tmp/data/target_stage1_small_big_mix",
    #                     help="path of the dataset")

    parser.add_argument("--netG", type=str, default=r'D:\Data\pix2pix_exp\model_pix2pix_unet\generator_78.pth', help="netG path to load")
    parser.add_argument("--netD", type=str, default='', help="netD path to load")

    parser.add_argument("--batch_size", type=int, default=3, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    parser.add_argument("--dataset_path", type=str, default=r"D:\Data\ele_img_big_small_mix_GAN_test",
                        help="path of the dataset")

    opt = parser.parse_args()
    print(opt)



    cuda = True if torch.cuda.is_available() else False


    # Initialize generator and discriminator
    generator = GeneratorUNet(in_channels=opt.channels_in, out_channels=opt.channels_out)

    if len(opt.netG) > 1:
        if cuda:
            generator.load_state_dict(torch.load(opt.netG), strict=True)
        else:
            generator.load_state_dict(torch.load(opt.netG, map_location=torch.device('cpu')), strict=True)
    else:
        print('no available netG path:{}'.format(opt.netG))

    if cuda:
        generator = generator.cuda()


    dataloader = ImageDataset_FMI(opt.dataset_path, x_l=opt.img_size_x, y_l=opt.img_size_x)
    dataloader = DataLoader(dataloader, shuffle=False, batch_size=opt.batch_size, drop_last=True, pin_memory=True,
                            num_workers=opt.n_cpu)


    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    prev_time = time.time()

    dyna_M = []
    stat_M = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Model inputs
            real_A = Variable(batch["B"].type(Tensor))
            real_B = Variable(batch["A"].type(Tensor))


            # GAN loss
            gen_parts = generator(real_A).cpu().detach().numpy()
            imgs = real_B.cpu().detach().numpy()
            mask = real_A.cpu().detach().numpy()

            # print(gen_parts.shape)  # torch.Size([3, 2, 128, 128])
            for j in range(gen_parts.shape[0]):
                # 动态图像的效果对比
                PSNR, SSIM, mse, rmse, mae, r2, Entropy_org, Entropy_vice, Con_org, Con_vice = cal_pic_generate_effect(
                    (imgs[j, 0, :, :]*256).astype(np.uint8), (gen_parts[j, 0, :, :]*256).astype(np.uint8))
                dyna_M.append([PSNR, SSIM, mse, rmse, mae, r2, Entropy_org, Entropy_vice, Con_org, Con_vice])
                print('dyna - PSNR:{:.4f} SSIM:{:.4f} mse:{:.4f} rmse:{:.4f} mae:{:.4f} '
                      'r2:{:.4f} Entropy:{:.4f},{:.4f} Contrast:{:.4f},{:.4f}'.format(
                    PSNR, SSIM, mse, rmse, mae, r2, Entropy_org, Entropy_vice, Con_org, Con_vice))

                # 静态图像的效果对比
                PSNR, SSIM, mse, rmse, mae, r2, Entropy_org, Entropy_vice, Con_org, Con_vice = cal_pic_generate_effect(
                    (imgs[j, 1, :, :]*256).astype(np.uint8), (gen_parts[j, 1, :, :]*256).astype(np.uint8))
                stat_M.append([PSNR, SSIM, mse, rmse, mae, r2, Entropy_org, Entropy_vice, Con_org, Con_vice])
                print('stat - PSNR:{:.4f} SSIM:{:.4f} mse:{:.4f} rmse:{:.4f} mae:{:.4f} '
                      'r2:{:.4f} Entropy:{:.4f},{:.4f} Contrast:{:.4f},{:.4f}'.format(
                    PSNR, SSIM, mse, rmse, mae, r2, Entropy_org, Entropy_vice, Con_org, Con_vice))

                show_Pic([1-imgs[j, 0, :, :], 1-gen_parts[j, 0, :, :], 1-imgs[j, 1, :, :], 1-gen_parts[j, 1, :, :],
                          mask[j, 0, :, :], mask[j, 1, :, :]], pic_order='32')

        dyna_M = np.array(dyna_M)
        stat_M = np.array(stat_M)
        M_all = []
        for i in range(dyna_M.shape[1]):
            M_Temp = np.mean(dyna_M[:, i])
            M_all.append(M_Temp)
            M_Temp = np.mean(stat_M[:, i])
            M_all.append(M_Temp)

        print(M_all)
