import torch.nn as nn
import torch.nn.functional as F
import torch


class FMI_Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(FMI_Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        # for out_filters, stride, normalize in [(64, 1, False), (128, 1, True), (256, 2, True), (512, 2, True)]:
        for out_filters, stride, normalize in [(32, 1, False), (64, 1, True), (128, 2, True), (256, 2, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

# a = FMI_Discriminator()
# print(a)