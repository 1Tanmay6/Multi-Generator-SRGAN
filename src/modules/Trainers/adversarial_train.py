import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from ..Utils import get_config_files, print_progress
from ..Data.custom_dataset_generator import CustomDatasetGenerator
from ..LossUtils import VGGFeatureExtractor
from ..Generator.generator import Generator
from ..Discriminator.discriminator import Discriminator


class AdversarialTrainer:
    def __init__(self, generator: Generator, discriminator: Discriminator) -> None:
        adv_config = get_config_files(key='adversarial_train')
        self.save_images = bool(adv_config['save_images'])
        if self.save_images:
            self.image_save_path = adv_config['image_save_path']
            os.makedirs(self.image_save_path, exist_ok=True)

        self.lr_D = float(adv_config['lr_D'])
        self.lr_G = float(adv_config['lr_G'])
        self.d_steps = int(adv_config['d_steps'])
        self.g_steps = int(adv_config['g_steps'])
        self.num_epochs = int(adv_config['num_epochs'])

        self.adversarial_loss = nn.BCELoss()
        self.content_loss = nn.MSELoss()

        self.generator = generator.cuda()
        self.discriminator = discriminator.cuda()
        self.feature_extractor = VGGFeatureExtractor().cuda()

        self.gen_losses = []
        self.dis_losses = []
        self.real_losses = []
        self.fake_losses = []
        self.g_losses = []
        self.content_losses = []
        self.epoch_wise_g_losses = []
        self.epoch_wise_d_losses = []

        self.optimizer_G = torch.optim.Adam(
            generator.parameters(), lr=self.lr_G)
        self.optimizer_D = torch.optim.Adam(
            discriminator.parameters(), lr=self.lr_D)

    def __call__(self, train_loader):
        self.train(train_loader=train_loader)

    def __str__(self) -> str:
        return "AdversarialTrainer class for training the generator and discriminator models"

    def __repr__(self) -> str:
        return "AdversarialTrainer(generator=Generator(), discriminator=Discriminator())"

    def train(self, train_loader: CustomDatasetGenerator):
        for epoch in range(self.num_epochs):
            for i, (imgs_lr, imgs_hr) in enumerate(train_loader):
                try:
                    imgs_lr = imgs_lr.cuda()
                    imgs_hr = imgs_hr.cuda()

                    for _ in range(self.d_steps):
                        self.optimizer_D.zero_grad()
                        gen_hr = self.generator(imgs_lr)
                        valid = torch.ones((imgs_lr.size(0), 1),
                                           requires_grad=False).cuda() * 0.9
                        fake = torch.zeros((imgs_lr.size(0), 1),
                                           requires_grad=False).cuda() * 0.1
                        real_loss = self.adversarial_loss(
                            self.discriminator(imgs_hr), valid)
                        fake_loss = self.adversarial_loss(
                            self.discriminator(gen_hr.detach()), fake)
                        d_loss = (real_loss + fake_loss) / 2

                        self.real_losses.append(real_loss.item())
                        self.fake_losses.append(fake_loss.item())
                        self.dis_losses.append(d_loss.item())

                        d_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.discriminator.parameters(), max_norm=1.0)
                        self.optimizer_D.step()

                    for index in range(self.g_steps):
                        self.optimizer_G.zero_grad()
                        gen_hr = self.generator(imgs_lr)
                        valid = torch.ones((imgs_lr.size(0), 1),
                                           requires_grad=False).cuda() * 0.9
                        g_loss = self.adversarial_loss(
                            self.discriminator(gen_hr), valid)
                        gen_features = self.feature_extractor(gen_hr)
                        real_features = self.feature_extractor(imgs_hr)
                        content_loss_value = self.content_loss(
                            gen_features, real_features)
                        g_total_loss = g_loss + content_loss_value
                        if index % self.g_steps == 0:
                            self.g_losses.append(g_loss.item())
                            self.content_losses.append(
                                content_loss_value.item())
                            self.gen_losses.append(g_total_loss.item())

                        g_total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.generator.parameters(), max_norm=1.0)
                        self.optimizer_G.step()
                        print_progress(epoch+1, self.num_epochs, i,
                                       len(train_loader), (d_loss.item(), g_total_loss.item()), adversarial_loss=True)
                    self.epoch_wise_g_losses.append(g_total_loss.item())
                    self.epoch_wise_d_losses.append(d_loss.item())

                except RuntimeError as e:
                    print(f"Error in batch {i}: {e}")
                    break
            if self.save_images:
                save_image(
                    gen_hr.data, f'{self.image_save_path}/gen_hr_epoch_{epoch}.png', normalize=True)
                save_image(
                    imgs_lr, f'{self.image_save_path}/gen_lr_epoch_{epoch}.png', normalize=True)

    def get_models(self):
        return self.generator, self.discriminator

    def get_losses(self):
        return self.gen_losses, self.dis_losses, self.real_losses, self.fake_losses, self.g_losses, self.content_losses, self.epoch_wise_g_losses, self.epoch_wise_d_losses

    def get_optimizers(self):
        return self.optimizer_G, self.optimizer_D

    def get_criteria(self):
        return self.adversarial_loss, self.content_loss

    def save_models(self, path):
        torch.save(self.generator.state_dict(),
                   os.path.join(path, 'generator.pth'))
        torch.save(self.discriminator.state_dict(),
                   os.path.join(path, 'discriminator.pth'))

    def save_models_script(self, path):
        model_scripted = torch.jit.script(self.generator)
        model_scripted.save(os.path.join(path, 'generator.pt'))
        model_scripted = torch.jit.script(self.discriminator)
        model_scripted.save(os.path.join(path, 'discriminator.pt'))


if __name__ == '__main__':
    dataset = CustomDatasetGenerator()
    train_loader, _ = dataset.dataset_generator()
    g = Generator()
    d = Discriminator()
    adv = AdversarialTrainer(g, d)
    adv.train(train_loader)
    print("Training done")
    torch.save(g.state_dict(), 'generator.pth')
    torch.save(d.state_dict(), 'discriminator.pth')
    print("Models saved")
    print("Exiting...")
    exit(0)
