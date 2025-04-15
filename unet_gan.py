import torch
import torch.nn as nn
import torch.optim as optim
import os
import torchvision.models as models
from torchvision.models import vgg16
from utils import visualize_sample_grid, plot_losses_live
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
# -------- Attention Block -------- #
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# -------- Hybrid U-Net with Attention -------- #
class HybridUNetGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=2, features=64):
        super(HybridUNetGenerator, self).__init__()
        resnet = models.resnet34(pretrained=True)
        self.layer0 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(resnet.children())[1:4]
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Decoder with Attention
        self.up4 = self._upsample_block(512, 256)
        self.att4 = AttentionBlock(256, 256, 128)
        self.up3 = self._upsample_block(512, 128)
        self.att3 = AttentionBlock(128, 128, 64)
        self.up2 = self._upsample_block(256, 64)
        self.att2 = AttentionBlock(64, 64, 32)
        self.up1 = self._upsample_block(128, 64)

        self.final = nn.ConvTranspose2d(64, output_channels, kernel_size=2, stride=2)
        self.tanh = nn.Tanh()

    def _upsample_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        e0 = self.layer0(x)
        e1 = self.layer1(e0)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        d4 = self.up4(e4)
        d4 = torch.cat([self.att4(g=d4, x=e3), d4], dim=1)

        d3 = self.up3(d4)
        d3 = torch.cat([self.att3(g=d3, x=e2), d3], dim=1)

        d2 = self.up2(d3)
        d2 = torch.cat([self.att2(g=d2, x=e1), d2], dim=1)

        d1 = self.up1(d2)

        return self.tanh(self.final(d1))

# -------- VGG16 Feature Extractor for Perceptual Loss -------- #
class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg_features = models.vgg16(pretrained=True).features
        self.slices = nn.ModuleList([
            vgg_features[:4],   # conv1_2
            vgg_features[4:9],  # conv2_2
            vgg_features[9:16], # conv3_3
            vgg_features[16:23] # conv4_3
        ])
        for param in self.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss()
        self.resize = resize

    def forward(self, x, y):
        # Resize to 224x224 if required
        if self.resize:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            y = nn.functional.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)

        loss = 0.0
        for slice_layer in self.slices:
            x = slice_layer(x)
            y = slice_layer(y)
            loss += self.criterion(x, y)
        return loss

# -------- PatchGAN Discriminator -------- #
class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(PatchDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)

# -------- Training Loop -------- #
def train_colorization_gan(generator, discriminator, train_loader, val_loader, epochs, device,
                           g_optimizer, d_optimizer, l1_loss_fn, bce_loss_fn,
                           save_dir, save_samples_every=5, enable_visualization=True,
                           samples_dir=None):
    perceptual_loss_fn = VGGPerceptualLoss().to(device)
    generator_losses = []
    discriminator_losses = []
    best_ssim = -1

    if samples_dir is None:
        samples_dir = os.path.join(save_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        generator.train()
        disc_loss_total = 0
        gen_loss_total = 0

        for L, AB in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            L, AB = L.to(device), AB.to(device)
            real_images = torch.cat([L, AB], dim=1)

            fake_AB = generator(L)
            fake_images = torch.cat([L, fake_AB.detach()], dim=1)

            # Discriminator
            real_output = discriminator(real_images)
            fake_output = discriminator(fake_images)
            real_labels = torch.ones_like(real_output)
            fake_labels = torch.zeros_like(fake_output)
            d_loss = (bce_loss_fn(real_output, real_labels) + bce_loss_fn(fake_output, fake_labels)) / 2

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Generator
            fake_images = torch.cat([L, fake_AB], dim=1)
            pred_fake = discriminator(fake_images)
            g_adv = bce_loss_fn(pred_fake, real_labels)
            g_l1 = l1_loss_fn(fake_AB, AB) * 100
            fake_LAB = torch.cat([L, fake_AB], dim=1)
            real_LAB = torch.cat([L, AB], dim=1)
            g_perceptual = perceptual_loss_fn(fake_LAB, real_LAB) * 10
            g_loss = g_adv + g_l1 + g_perceptual

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            disc_loss_total += d_loss.item()
            gen_loss_total += g_loss.item()

        avg_g = gen_loss_total / len(train_loader)
        avg_d = disc_loss_total / len(train_loader)
        generator_losses.append(avg_g)
        discriminator_losses.append(avg_d)
        print(f"Epoch {epoch} | G Loss: {avg_g:.4f} | D Loss: {avg_d:.4f}")

        # ---------------- Validation ---------------- #
        generator.eval()
        total_ssim, total_psnr, count = 0, 0, 0
        with torch.no_grad():
            for L, AB in val_loader:
                L, AB = L.to(device), AB.to(device)
                fake_AB = generator(L)

                fake_imgs = torch.cat([L, fake_AB], dim=1).cpu()
                real_imgs = torch.cat([L, AB], dim=1).cpu()

                for f_img, r_img in zip(fake_imgs, real_imgs):
                    f_np = f_img.permute(1, 2, 0).numpy()
                    r_np = r_img.permute(1, 2, 0).numpy()
                    f_np = (f_np + 1) / 2
                    r_np = (r_np + 1) / 2
                    f_np = cv2.resize(f_np, (256, 256))
                    r_np = cv2.resize(r_np, (256, 256))

                    score, _ = ssim(f_np, r_np, channel_axis=-1, full=True, data_range=1.0)
                    total_ssim += score
                    total_psnr += psnr(r_np, f_np, data_range=1.0)
                    count += 1

        avg_ssim = total_ssim / count
        avg_psnr = total_psnr / count
        print(f"Validation SSIM: {avg_ssim:.4f} | PSNR: {avg_psnr:.2f}")

        # Save best generator based on SSIM
        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            torch.save(generator.state_dict(), os.path.join(save_dir, "best_generator.pth"))
            print(" Saved Best Generator (based on SSIM)")

        # Save samples every few epochs
        if epoch % save_samples_every == 0:
            generator.eval()
            with torch.no_grad():
                L, AB = next(iter(val_loader))
                L, AB = L.to(device), AB.to(device)
                fake_AB = generator(L)
                samples = torch.cat([L, fake_AB], dim=1)
                sample_grid = visualize_sample_grid(samples, unnormalize=True)
                save_path = os.path.join(samples_dir, f"epoch_{epoch}.png")
                plt.imsave(save_path, sample_grid)
                print(f" Saved sample at: {save_path}")

        # Optional: live loss plot
        if enable_visualization:
            plot_losses_live(generator_losses, discriminator_losses)