import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable

from torchvision import transforms, datasets
from torchvision import utils
from torchvision.utils import save_image
import torchvision.utils as vutils

from net import Generator, Discriminator

image_size = 1024
device = 'cuda'
nz = 100
ngf = 64
ndf = 64
nc = 3

def create_transform():
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    return  transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def set_dataset(dataset_path, transform, batch_size):
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def run_train(netD, netG, dataset, options):
    netG.train()
    netD.train()
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(options["batch_size"], 64, 1, 1, device=device)
    print(netG)
    print(netD)
    
    # 本物
    real_label = 1
    # 偽物
    fake_label = 0

    optimizerG = optim.Adam(netG.parameters(),
                             lr=options["lr"],
                             betas=(0.5, 0.999))

    optimizerD = optim.Adam(netD.parameters(),
                             lr=options["lr"],
                             betas=(0.5, 0.999))

    with open('log.txt', 'w') as f:
        D_runnign_loss = 0
        G_runnign_loss = 0

        for epoch in range(options['epoch']):
            try:
                print(f'{epoch + 1}')
                for data in dataset:

                    # 勾配の更新
                    optimizerD.zero_grad()

                    raw_image, raw_label = data
                    real_image = raw_image.to('cuda')
                    label = raw_label.to('cuda')

                    errorD_real = 0
                    errorD_fake = 0
                    errorG = 0

                    batch_size = 0
                    error_real_image = 0
                    error_discriminator = 0
                    fake = None
                    real_image = None
                    # 本物に近似しているか
                    # Train with Real
                    netD.zero_grad()
                    raw_image, raw_label = data
                    real_image = raw_image.to(device)
                    label = raw_label.to(device)
                    label = torch.full((options["batch_size"],),
                                        real_label, device=device)

                    try:
                        real_image_output = netD(real_image)
                        print(real_image_output)
                        error_real_image = criterion(real_image_output,
                                                    label)
                        error_real_image.backward()
                        D_x = real_image_output.mean().item()
                    except Exception as e:
                        print("Pass")
                        print("Real Image")
                        print(str(e))
                    print("Begin")
                    try:
                        noise = torch.randn(options["batch_size"], nz, 1, 1, device=device)
                        fake = netG(noise)
                        print(fake)
                        label.fill_(fake_label)
                        output = netD(fake.detach())
                        error_discriminator_fake = criterion(output, label)
                        error_discriminator_fake.backward()
                        D_G_z1 = output.mean().item()
                        error_discriminator = error_real_image + error_discriminator_fake
                        optimizerD.step()
                    except Exception as e:
                        print("Pass")
                        print(str(e))
                        print("Fake Image")
                        print()
                    print("END")

                    try:
                        # Update Network
                        netG.zero_grad()
                        label.fill_(real_label) 
                        output = netD(fake) # 鑑定を行う
                        errorG = criterion(output, label)
                        errorG.backward()
                        D_G_z2 = output.mean().item()
                        optimizerG.step()
                    except Exception as e:
                        print("Pass")
                        print(str(e))
                        print("KANTEI")
                        print()
                    print(netG)

                    fake = None
                    vutils.save_image(real_image_output,
                                      'real_samples.png',
                                      normalize=True)
                    fake = netG(fixed_noise)
                    vutils.save_image(fake.detach(),
                                      'fake_samples_epoch_%03d.png' %
                                      (epoch), normalize=True)

                    save_image(netG.state.dict, f'{epoch}.pth')
                    save_image(netD.state.dict, f'{epoch}.pth')
            except Exception as e:
                print(f"Pass:[{epoch}] {e}\n")
                print()
                pass

def main():
    train_dataset_path = 'D:\project\dcgan2\dataset'
    # train_dataset_path = 'D:\project\dataset\\food'
    options = {
        'batch_size': 16,
        'z_dim': 16,
        'epoch': 10,
        'lr': 2e-3
    }

    data_transform = create_transform()
    train_dataset = set_dataset(train_dataset_path, data_transform,
                                options['batch_size'])

    G = Generator().to('cuda')
    D = Discriminator().to('cuda')

    run_train(D, G, train_dataset, options)

main() if __name__ == '__main__' else None